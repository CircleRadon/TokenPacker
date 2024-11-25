
import copy
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from llava.train.train import preprocess, preprocess_multimodal
from llava.patch_divide import Image_Patch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

class CustomDataset(Dataset):

    def __init__(self,
                 tokenizer=None,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_gt_per_img = max_gt_per_img
        self.img_prefix = img_prefix
        self.data_infos = self.load_annotations(ann_file)

        self.image_patch = Image_Patch()
        self.preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

        super().__init__()

    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue

            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    def get_ann_info(self, idx):

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info
    
    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

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
            question = '<region>'
            question = question.replace('<region>', '<mask>')
            if i == 0:
                question = self.begin_str + question
            answer = ori_labels[i]
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        # a hard code [] for sources
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
        data_dict['h_block'] = h_block
        data_dict['w_block'] = w_block
        data_dict['masks'] = ori_masks
        return data_dict

    def read_process_image(self, img_path):

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

        return image_tensor, h_block, w_block, img_h_, img_w_
    
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path =os.path.join(self.img_prefix, data_info['filename'])
        image, h_block, w_block, img_h_, img_w_ = self.read_process_image(img_path)
        block_size = 336 #fixed
        
        gt_masks = []
        gt_labels = []
        for ann in ann_info:
            mask_ori = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            mask_tensor = torch.from_numpy(mask_ori).unsqueeze(dim=0)
            mask_inter = F.interpolate(mask_tensor.unsqueeze(dim=0), size=(img_h_,img_w_), mode='nearest').squeeze(dim=0).squeeze(dim=0)
            mask = torch.zeros((block_size*h_block, block_size*w_block)).to(dtype=mask_inter.dtype)
            mask[:img_h_, :img_w_] = mask_inter
            gt_masks.append(mask)

            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels,
            h_block = h_block,
            w_block = w_block
        )
        return data_item

    def __getitem__(self, idx):

        data_item = self.get_data_item(idx)
        data_dict = self.process_text(data_item=data_item)

        return data_dict

class COCODataset(CustomDataset):

    def __init__(self,
                 tokenizer=None,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        self.begin_str = '<image>\nIn the conversation below, you simply answer the category name based on what you see ' \
                        'in the imagery inside a particular region. I will give you only one region each time.\n' 

class PartImagenet(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )

        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )

        begin_str = '<image>\nIn the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'

class PascalPart(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        CAT_CLASSES = ('potted plant', 'aeroplane', 'cow', 'cat', 'bus', 'horse', 'car', 
                    'dog', 'bicycle', 'person', 'bird', 'bottle', 'sheep', 'motorbike')

        SUB_CLASSES = ('eye', 'window', 'cap', 'headlight', 'hand', 'mirror', 'arm', 'plant', 
                    'wheel', 'ear', 'pot', 'foot', 'leg', 'nose', 'body', 'horn', 'handlebar', 
                    'neck', 'license plate', 'paw', 'saddle', 'head', 'muzzle', 'tail', 'wing', 
                    'beak', 'hair', 'torso', 'door', 'mouth')

        begin_str = '<image>\n In the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category:subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'

class RefCOCO(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)

        self.begin_str = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'

    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue
            
            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
        
    def get_data_item(self, idx):

        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path =os.path.join(self.img_prefix, data_info['filename'])
        image, h_block, w_block, img_h_, img_w_ = self.read_process_image(img_path)

        block_size = 336
        gt_masks = []
        gt_labels = []
        for ann in ann_info:
        
            mask_ori = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            mask_tensor = torch.from_numpy(mask_ori).unsqueeze(dim=0)
            mask_inter = F.interpolate(mask_tensor.unsqueeze(dim=0), size=(img_h_,img_w_), mode='nearest').squeeze(dim=0).squeeze(dim=0)
            mask = torch.zeros((block_size*h_block, block_size*w_block)).to(dtype=mask_inter.dtype)
            mask[:img_h_, :img_w_] = mask_inter

            gt_masks.append(mask)
            
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(data_info['caption'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels,
            h_block = h_block,
            w_block = w_block
        )
        return data_item

class RefCOCOP(RefCOCO):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        self.begin_str = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image and its basic attibuts, you should not ' \
                         'give its position within the image.'                        
