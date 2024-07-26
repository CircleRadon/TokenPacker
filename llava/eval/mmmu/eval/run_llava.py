import torch
import os
import random

import numpy as np
import math
from tqdm import tqdm
import json

from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils_ind import call_llava_engine_df
from utils.eval_utils import evaluate, parse_multi_choice_response, parse_open_response
import torch.nn.functional as F
from functools import partial
from llava.patch_divide import Image_Patch
from torchvision.transforms import Compose, ToTensor, Normalize

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main():
    parser = ArgumentParser()
    # parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
    #                     help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_8bit', type=bool, default=False)

    args = parser.parse_args()
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

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


    # run for each subject
    sub_dataset_list = []
    subjects = [x for x in CAT_SHORT2LONG.values()]
    '''
    subjects = [
        'Architecture_and_Engineering', 'Computer_Science', 'Electronics',
        'Energy_and_Power', 'Materials', 'Mechanical_Engineering'
    ]
    '''
    for subject in tqdm(subjects):
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    sub_dataset_list = get_chunk(sub_dataset_list, args.num_chunks, args.chunk_idx)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # samples = []
    out_samples = []
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample['image']:
            image = sample['image'].convert('RGB')
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

            sample['image'] = image_tensor
            
        # samples.append(sample)
        mode = model.config.image_aspect_ratio
        with torch.no_grad():
            response = call_model_engine(args, sample, model, tokenizer, processor, h_block, w_block, mode)
            if sample['question_type'] == 'multiple-choice':
                parsed_pred = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
                out_sample = {
                    'id': sample['id'],
                    'question_type': sample['question_type'],
                    'answer': sample['answer'],
                    'response': response,
                    'parsed_pred': parsed_pred,
                    'index2ans': sample['index2ans'],
                }
            else:  # open question
                parsed_pred = parse_open_response(response)
                out_sample = {
                    'id': sample['id'],
                    'question_type': sample['question_type'],
                    'answer': sample['answer'],
                    'response': response,
                    'parsed_pred': parsed_pred,
                }
            out_samples.append(out_sample)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, sample in enumerate(out_samples):
        ans_file.write(json.dumps(sample) + "\n")
    ans_file.close()

if __name__ == '__main__':
    main()

