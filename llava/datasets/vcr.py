"""
This code is largely based on https://github.com/jshilong/GPT4RoI/blob/main/gpt4roi/datasets/vcr.py
"""
import copy
import json
import os
import random
from tkinter import N

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import path
from matplotlib import pyplot as plt
from llava.train.train import preprocess, preprocess_multimodal
from torchvision.transforms import Compose, ToTensor, Normalize
from llava.patch_divide import Image_Patch

WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]

Ref_WAY = [
    'There are <region> in the image,',
    'There are some regions <region>,',
    'Given <region>,',
    'Given <region> in the image,',
    '<region>,',
    'Several regions <region> are in the image,',
    '<region> in the given image,'
]

def _spaced_points(low, high,n):
    """ We want n points between low and high, but we don't want them to touch either side"""
    padding = (high-low)/(n*2)
    return np.linspace(low + padding, high-padding, num=n)

def make_mask(height, width, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2, conf.]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((height, width), dtype=np.bool_)
    
    xy = np.meshgrid(_spaced_points(box[0], box[2], n=width),
                     _spaced_points(box[1], box[3], n=height)) 
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((height, width))
    return mask.astype(np.float32)

class VCRDataset(Dataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,

                 ):
        super(VCRDataset, self).__init__()


        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.image_patch = Image_Patch()
        self.preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

        self.begin_str = """<image>.\nThis provides an overview of the picture.\n"""
        self.data_infos = self.load_annotations(ann_file)
        print('normal_vcr', len(self.data_infos))

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_list = [json.loads(line) for line in f]
        data_infos = []

        import re

        def replace_numbers_with_tags(s, class_names):
            pattern = r'\b(\d+)\b'
            try:
                result = re.sub(pattern, lambda match: f'{class_names[int(match.group(1))]} at region{match.group(1)}', s)
            except:
                # contain number not for instance
                return None
            return result


        for ann in ann_list:

            metadata_fn_path = ann['metadata_fn']
            img_fn = ann['img_fn']
            img_path = os.path.join(self.img_prefix,img_fn)
            annotations = json.load(open(os.path.join(self.img_prefix, metadata_fn_path)))
            masks = annotations['segms']
            bboxes = np.array(annotations['boxes'])

            class_names = ann['objects']
            num_objects = len(class_names)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <mask>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = random.choice(Ref_WAY)

            begion_string = ref_prefix.replace('<region>', ref_string)
            qa_s = []

            q = ann['question_orig']
            q = replace_numbers_with_tags(q, class_names)
            a = ann['answer_orig']
            a = replace_numbers_with_tags(a, class_names)
            why = ann['rationale_orig']
            why = replace_numbers_with_tags(why, class_names)
            if (q is None) or (a is None) or (why) is None:
                continue

            qa_s.append({'from': 'human', 'value': begion_string + q})
            qa_s.append({'from': 'gpt', 'value': a})
            qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})
            qa_s.append({'from': 'gpt', 'value': why})

            data_infos.append(dict(
                img_path = img_path,
                bboxes = bboxes,
                masks = masks,
                labels= class_names,
                qas = qa_s)
            )

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i):
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        masks = data_info['masks']
        bboxes = data_info['bboxes']

        qas = data_info['qas']
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)

        image = image.unsqueeze(0)
        h, w = image.shape[-2:]
        block_size = 336
        h_block, w_block = self.image_patch.calculate(h, w)
        h_ratio = block_size*h_block/h
        w_ratio = block_size*w_block/w
        if h_ratio<=w_ratio:
            img_w_ = min(block_size*w_block, round(w*h_ratio))
            img_h_ = block_size*h_block
        else:
            img_w_ = block_size*w_block
            img_h_ = min(block_size*h_block, round(h*w_ratio))
        image_inter = F.interpolate(image, size=(img_h_,img_w_), mode='bilinear')
        image = torch.zeros((1, 3, block_size*h_block, block_size*w_block)).to(dtype=image_inter.dtype, device=image_inter.device)
        image[:, :, :img_h_, :img_w_] = image_inter

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
        block_size = 336 

        gt_masks = []
        for i,mask in enumerate(masks):

            pred_mask = np.zeros((h, w))
            int_box =  [round(box) for box in bboxes[i][:-1]]
            height_ = int(int_box[3]-int_box[1])
            width_ = int(int_box[2]-int_box[0])
            box_mask = make_mask(height_, width_, bboxes[i], mask)
            pred_mask[int_box[1]:int_box[3], int_box[0]:int_box[2]] = box_mask

            mask_tensor = torch.from_numpy(pred_mask).unsqueeze(dim=0)
            mask_inter = F.interpolate(mask_tensor.unsqueeze(dim=0), size=(img_h_,img_w_), mode='nearest').squeeze(dim=0).squeeze(dim=0)
            mask = torch.zeros((block_size*h_block, block_size*w_block)).to(dtype=mask_inter.dtype)
            mask[:img_h_, :img_w_] = mask_inter
            gt_masks.append(mask)

        ori_masks = np.array(gt_masks)
        ori_masks = torch.from_numpy(ori_masks) 

        qas = copy.deepcopy(qas)
        qas[0]['value'] = self.begin_str + qas[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image_tensor
        data_dict['masks'] = ori_masks

        data_dict['h_block'] = h_block
        data_dict['w_block'] = w_block

        return data_dict
