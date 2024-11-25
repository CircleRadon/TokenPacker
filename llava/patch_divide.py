import torch
from torchvision.ops.boxes import box_area


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
    def __init__(self, image_size=336, anchors=[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)
        ]):
  
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        self.anchor_list = anchors

        self.anchors = torch.tensor(
            [[0, 0, _[0]*image_size[0], _[1]*image_size[1]] 
            for _ in anchors], requires_grad=False
        )
        
        self.anchor_areas = box_area(self.anchors)
                
    def calculate(self, h, w):
        input_box = torch.tensor([0, 0, h, w]).unsqueeze(0)
        ratio = self.anchors[:, 2:]/input_box[:, 2:]
        ratio = ratio.min(dim=-1)[0]
        score = torch.round(h*ratio) * torch.round(w*ratio) / self.anchor_areas
        iou, _ = box_iou(self.anchors, self.anchor_areas, input_box*1.4)
        iou = iou[:, 0]
        score = score + iou*0.1
        idx = torch.argmax(score)
        return self.anchor_list[idx]