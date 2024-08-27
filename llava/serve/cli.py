import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
import torch.nn.functional as F
from functools import partial
from llava.patch_divide import Image_Patch
from torchvision.transforms import Compose, ToTensor, Normalize

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from functools import partial
import time

def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length = 2048,
        padding_side="right",
        use_fast = True
    )
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,   
        torch_dtype=torch.bfloat16,
    ).cuda()

    for m in model.modules():
        m.tokenizer = tokenizer

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    patch_num = getattr(model.config, 'patch_num', '9')
    image_patch = Image_Patch(int(patch_num))
    preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

    
    while True:
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image_file = input("image file: ")

        image = Image.open(image_file).convert('RGB')

        if model.config.image_aspect_ratio == 'slice':
            image = preprocess(image)
            image = image.unsqueeze(0)
            h, w = image.shape[-2:]
            block_size = 336
            h_block, w_block = image_patch.calculate(h, w)
            h_ratio = block_size*h_block/h
            w_ratio = block_size*w_block/w
            if h_ratio<=w_ratio:
                w_ = min(block_size*w_block, round(w*h_ratio))
                h_ = block_size*h_block
            else:
                w_ = block_size*w_block
                h_ = min(block_size*h_block, round(h*w_ratio))
            image_inter = F.interpolate(image, size=(h_,w_), mode='bilinear')
            image = torch.zeros((1, 3, block_size*h_block, block_size*w_block)).to(dtype=image_inter.dtype, device=image_inter.device)
            image[:, :, :h_, :w_] = image_inter

            split_images = []
            for i_ in range(h_block):
                for j_ in range(w_block):
                    image_s = image[:,:,block_size*i_:block_size*(i_+1), block_size*j_:block_size*(j_+1)]
                    split_images.append(image_s)
            if len(split_images)>1:
                h_ratio = block_size/h
                w_ratio = block_size/w
                if h_ratio<=w_ratio:
                    w_ = min(block_size, round(w*h_ratio))
                    h_ = block_size
                else:
                    w_ = block_size
                    h_ = min(block_size, round(h*w_ratio))
                image_inter = F.interpolate(image, size=(h_,w_), mode='bilinear')
                image_s = torch.zeros((1, 3, block_size, block_size)).to(dtype=image_inter.dtype, device=image_inter.device)
                image_s[:, :, :h_, :w_] = image_inter
                split_images.append(image_s)
            image_tensor = torch.cat(split_images, dim=0)
        else:
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0)
            h_block = 1
            w_block = 1

        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        # inp = "what is in the image?"

        print(f"{roles[1]}: ", end="")

        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        mode = model.config.image_aspect_ratio
        with torch.inference_mode():
            model.orig_forward = model.forward
            model.forward = partial(model.orig_forward,
                                    mode=mode,
                                    h_block=h_block,
                                    w_block=w_block)
            start = time.time()

            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            model.forward = model.orig_forward

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        end = time.time()
        print("***time: ", end-start)
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path/to/tokenpacker")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
