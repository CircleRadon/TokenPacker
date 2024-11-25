import copy
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from .stage2_data import CustomDataset
from llava.train.train import preprocess, preprocess_multimodal

LIMIT = " Answer the question using a short phrase."
QUESTIONS =  [
    'Give me a short description of <region>.',
    'Can you give me a short description of <region>?',
    'Can you provide me with a short description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words?",
    'What can you tell me about the region indicated by <region> in the image in few words?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a concise description?",
    'Could you describe the region shown as <region> in the picture concisely?',
    'What can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a brief description of the region marked with <region> in the image.',
    'Can you give me a brief introduction of the region labeled as <region> in the picture?',
    "I'm interested in knowing the region represented by <region> in the photo. Can you describe it in several words?",
    'What is the region outlined by <region> in the picture like? Could you give me a streamlined description?',
    'Can you provide me with a brief description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words, please?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a simple description?",
    'Could you describe the region shown as <region> in the picture in several words?',
    'Please provide me with a simple description of the region marked with <region> in the image, please.',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in few words, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a simple and clear description?',
    'Please describe the region <region> in the image concisely.',
    'Can you offer a simple analysis of the region <region> in the image?',
    'Could tell me something about the region highlighted by <region> in the picture briefly?',
    'Can you share a simple rundown of the region denoted by <region> in the presented image?'
]

class VGDATA(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=3,
                 ):

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.max_gt_per_img = max_gt_per_img

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)

        self.begin_str = """<image>\nThis provides an overview of the picture.\n"""


    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.img_prefix, data_info['filename'])
        image, h_block, w_block, img_h_, img_w_ = self.read_process_image(img_path)

        block_size = 336 
        gt_labels = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            mask_tensor = torch.from_numpy(mask).unsqueeze(dim=0)
            mask_inter = F.interpolate(mask_tensor.unsqueeze(dim=0), size=(img_h_,img_w_), mode='nearest').squeeze(dim=0).squeeze(dim=0)
            mask = torch.zeros((block_size*h_block, block_size*w_block)).to(dtype=mask_inter.dtype)
            mask[:img_h_, :img_w_] = mask_inter

            gt_labels.append(ann['caption'])
            gt_masks_ann.append(mask)

        data_item = dict(
            img = image,
            gt_labels=gt_labels,
            gt_masks=gt_masks_ann,
            h_block = h_block,
            w_block = w_block
        )
        return data_item

    
    def process_text(self, data_item):
        image = data_item['img']
        h_block = data_item['h_block']
        w_block = data_item['w_block']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])
        ori_masks = torch.from_numpy(ori_masks) 

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_masks = ori_masks[shuffle_ids]
        ori_labels = [ori_labels[i] for i in shuffle_ids]

        sources = dict()

        sources['conversations'] = []

        for i in range(len(ori_labels)):
            question = random.choice(QUESTIONS).strip()
            question = question.replace('<region>', '<mask>')
            if i == 0:
                question = self.begin_str + question
            question += LIMIT
            answer = ori_labels[i]
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        sources = preprocess_multimodal(
            copy.deepcopy([sources['conversations']]),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True
            )
        
        # get single
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = ori_masks

        data_dict['h_block'] = h_block
        data_dict['w_block'] = w_block
        
        return data_dict
