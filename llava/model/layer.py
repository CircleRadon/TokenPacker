import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskExtractor(nn.Module):
    def __init__(self, mask_shape=112, embed_dim=1024, out_dim=4096):
        super(MaskExtractor, self).__init__()
        self.mask_shape = mask_shape
        self.mask_pooling = MaskPooling()
        self.feat_linear = nn.Linear(embed_dim, out_dim)
        self.num_level = 4
        self.res_layers = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(4)])

    def forward(self, feats, masks, h_block, w_block):
        query_feats = []
        pos_feats = []
        num_imgs = len(masks)

        block_size = 24
        img_idx = 0

        for idx in range(num_imgs):
            mask = masks[idx].unsqueeze(0).float()
            merge_feats = []
            for single_feat in feats:
                single_feat = single_feat.reshape(single_feat.shape[0], block_size, block_size, -1).permute(0,3,1,2)

                merge_feat = torch.zeros(1,1024,block_size*h_block[idx],block_size*w_block[idx]).to(dtype=single_feat.dtype, device=single_feat.device)
                idx_ = 0
                for i_ in range(h_block[idx]):
                    for j_ in range(w_block[idx]):
                        merge_feat[:,:,block_size*i_:block_size*(i_+1), block_size*j_:block_size*(j_+1)] = single_feat[img_idx+idx_:img_idx+idx_+1]
                        idx_+=1
                if h_block[idx]*w_block[idx]>1:
                    idx_+=1 # global
                merge_feats.append(merge_feat)

            img_idx += idx_
            mask_feats = mask.new_zeros(self.num_level, mask.shape[1], 1024)

            for i in range(self.num_level):
                feat = merge_feats[i]

                raw_dtype = feat.dtype
                feat = feat.to(mask.dtype)
                mask_feat_raw = self.mask_pooling(feat, mask)
                mask_feat_flatten = mask_feat_raw.reshape(-1, mask_feat_raw.shape[-1])

                if i==0:
                    res2 = self.res_layers[0].to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = res2(mask_feat_flatten)
                elif i==1:
                    res3 = self.res_layers[1].to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = res3(mask_feat_flatten)
                elif i==2:
                    res4 = self.res_layers[2].to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = res4(mask_feat_flatten)
                else:
                    res5 = self.res_layers[3].to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = res5(mask_feat_flatten)
                
                mask_feat = mask_feat.reshape(*mask_feat_raw.shape[:2], -1)
                mask_feat = mask_feat.to(raw_dtype)
                mask_feats[i] = mask_feat[0]

            mask_feats = mask_feats.sum(0)
            self.feat_linear = self.feat_linear.to(dtype=mask_feats.dtype, device=mask_feats.device)
            mask_feats_linear = self.feat_linear(mask_feats)
            query_feats.append(mask_feats_linear)

        return query_feats

    
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x
