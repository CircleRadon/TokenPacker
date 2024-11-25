import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import copy
import random
from .stage2_data import CustomDataset
from llava.train.train import preprocess, preprocess_multimodal
from llava.patch_divide import Image_Patch
import re
from torchvision.transforms import Compose, ToTensor, Normalize

DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]


class ConversationDataset(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.begin_str = "<image>\nThis provides an overview of the picture.\n"
        self.image_patch = Image_Patch()
        self.preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])
        super().__init__(tokenizer, data_args, ann_file, img_prefix)


    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            str_region = ""
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
                if i>0:
                    str_region+= ','
                str_region += "region"+str(i+1)+"<mask>"

            for i in range(len(ann['conversations'])//2):
                    
                if i==0:
                    if region_num==1:
                        mid_str = "There are 1 part region in the picture: "+str_region+'. '
                    else:
                        mid_str = "There are {} part regions in the picture: ".format(str(region_num))+str_region+'. '

                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    question = self.begin_str+mid_str+question
                    qa_s.append({'from': 'human', 'value': question+self.limit})         
                else:
                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    qa_s.append({'from': 'human', 'value': question+self.limit})         

                
                answer = ann['conversations'][i*2+1]['value']
                answer = answer.replace('<','').replace('>','')
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos

    def __getitem__(self, i):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        masks_raw = data_info['masks']
        qas = data_info['qas']

        image, h_block, w_block, img_h_, img_w_ = self.read_process_image(img_path)

        block_size = 336
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

        masks = []
        for mask_r in masks_raw:            
            mask_ori = self.annToMask(mask_r, height, width)
            mask_tensor = torch.from_numpy(mask_ori).unsqueeze(dim=0)
            mask_inter = F.interpolate(mask_tensor.unsqueeze(dim=0), size=(img_h_,img_w_), mode='nearest').squeeze(dim=0).squeeze(dim=0)
            mask = torch.zeros((block_size*h_block, block_size*w_block)).to(dtype=mask_inter.dtype)
            mask[:img_h_, :img_w_] = mask_inter

            masks.append(mask)
            
        masks = np.array(masks)

        data_dict['image'] = image
        data_dict['h_block'] = h_block
        data_dict['w_block'] = w_block
        data_dict['masks'] = torch.Tensor(masks)

        return data_dict

class OspreyPartLevel(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

class OspreyLVISPosNeg(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name']
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
        
            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                question = re.sub(r'<region\d+>', '<mask>', question)
                if i==0:
                    question = self.begin_str+question
                qa_s.append({'from': 'human', 'value': question})         
             
                answer = ann['conversations'][i*2+1]['value']
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))

        return data_infos

      

class OspreyConversations(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.limit = ""
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

class OspreyShortForm(ConversationDataset):
     def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

class OspreyDetailedDescription(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                question = random.choice(DETAILED_QUESTIONS)
                question = question.replace('<region>', '<mask>')
                if i==0:
                    qa_s.append({'from': 'human', 'value': self.begin_str+question})         
                else:
                    qa_s.append({'from': 'human', 'value': question})     
            
                answer = re.findall(r"<.*>:\ (.*)", ann['description'][i])[0]
           
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos
