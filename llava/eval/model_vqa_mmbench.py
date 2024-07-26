import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
import math
import torch.nn.functional as F
from functools import partial
from llava.patch_divide import Image_Patch
from torchvision.transforms import Compose, ToTensor, Normalize


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


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

    patch_num = getattr(model.config, 'patch_num', '9')
    image_patch = Image_Patch(patch_num=int(patch_num))
    preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

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

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            mode = model.config.image_aspect_ratio

            with torch.inference_mode():
                model.orig_forward = model.forward
                model.forward = partial(model.orig_forward,
                                    mode=mode,
                                    h_block = [h_block],
                                    w_block = [w_block]
                                    )
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.bfloat16().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
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
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
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
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
