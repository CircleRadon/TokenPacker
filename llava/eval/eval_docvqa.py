import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
from PIL import Image
import math
import torch.nn.functional as F
from functools import partial
from llava.patch_divide import Image_Patch
from torchvision.transforms import Compose, ToTensor, Normalize

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.patch_num = getattr(model_config, 'patch_num', '9')
        self.image_patch = Image_Patch(patch_num=int(self.patch_num))
        self.preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])


    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"].split('/')[-1]
        qs = line["question"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs = qs+'\nAnswer the question using a single word or phrase.'

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        if self.model_config.image_aspect_ratio == 'slice':
            image = self.preprocess(image)
            image = image.unsqueeze(0)
            h, w = image.shape[-2:]
            block_size = 336
            h_block, w_block = self.image_patch.calculate(h, w)
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
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            image_tensor = image_tensor.unsqueeze(0)
            h_block = 1
            w_block = 1

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, h_block, w_block

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
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

    questions = json.load(open(os.path.expanduser(args.question_file)))["data"]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, h_block, w_block), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["questionId"]

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            model.orig_forward = model.forward
            model.forward = partial(model.orig_forward,
                                    h_block=h_block,
                                    w_block=w_block
                                )

            output_ids = model.generate(
                input_ids,
                images=image_tensor.squeeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

            model.forward = model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"questionId": idx,
                                   "answer": outputs})+ "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
