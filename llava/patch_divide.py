import torch
from torchvision.ops.boxes import box_area

patches_9=[
    (1,1),
    (1,2),(2,1),
    (1,3),(3,1),
    (2,2),(1,4),(4,1),
    (1,5),(5,1),
    (1,6),(6,1),(2,3),(3,2),
    (1,7),(7,1),
    (4,2),(2,4),(1,8),(8,1),
    (3,3),(1,9),(9,1)
]

patches_16=[
    (1,1),
    (1,2),(2,1),
    (1,3),(3,1),
    (2,2),(1,4),(4,1),
    (1,5),(5,1),
    (1,6),(6,1),(2,3),(3,2),
    (1,7),(7,1),
    (4,2),(2,4),(1,8),(8,1),
    (3,3),(1,9),(9,1),
    (2,5),(5,2), 
    (2,6),(6,2),(3,4), (4,3),
    (2,7),(7,2),
    (3,5),(5,3),
    (2,8),(8,2),(4,4)
]

patches_25=[
    (1,1),
    (1,2),(2,1),
    (1,3),(3,1),
    (2,2),(1,4),(4,1),
    (1,5),(5,1),
    (1,6),(6,1),(2,3),(3,2),
    (1,7),(7,1),
    (4,2),(2,4),(1,8),(8,1),
    (3,3),(1,9),(9,1),
    (2,5),(5,2), 
    (2,6),(6,2),(3,4), (4,3),
    (2,7),(7,2),
    (3,5),(5,3),
    (2,8),(8,2),(4,4),
    (3,6),(6,3),(2,9),(9,2),
    (4,5),(5,4),(2,10),(10,2),
    (3,7),(7,3),
    (11,2),(2,11),
    (4,6),(6,4),(12,2),(2,12),(3,8),(8,3),(4,6),(6,4),
    (5,5)
]


def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+eps)
    return iou, union

class Image_Patch:
    def __init__(self, image_size=336, patch_num=9):
        if patch_num == 9:
            patches = patches_9
        elif patch_num == 16:
            patches = patches_16
        elif patch_num == 25:
            patches = patches_25
        else:
            raise(NotImplementedError)
  
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        self.patch_list = patches

        self.patches = torch.tensor(
            [[0, 0, _[0]*image_size[0], _[1]*image_size[1]] 
            for _ in patches], requires_grad=False
        )
        
        self.patch_areas = box_area(self.patches)
                
    def calculate(self, h, w):
        input_box = torch.tensor([0, 0, h, w]).unsqueeze(0)
        ratio = self.patches[:, 2:]/input_box[:, 2:]
        ratio = ratio.min(dim=-1)[0]
        score = torch.round(h*ratio) * torch.round(w*ratio) / self.patch_areas
        iou, _ = box_iou(self.patches, self.patch_areas, input_box*1.4)
        iou = iou[:, 0]
        score = score + iou*0.1
        idx = torch.argmax(score)
        return self.patch_list[idx]