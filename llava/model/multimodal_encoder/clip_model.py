from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import (CLIPEncoderLayer,CLIPVisionEmbeddings,BaseModelOutputWithPooling,
                                                    CLIPEncoder, CLIPVisionTransformer, CLIPVisionModel)
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
import torch

PATCH_NUM_WIDTH = 24
PATCH_NUM_HEIGHT = 24
MAX_PATCHES = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT
POSITION_EMBEDDING_LENGTH = 1024

class adapt_CLIPVisionModel(CLIPVisionModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = adapt_CLIPVisionTransformer(config)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        resized_patch_width = None, 
        resized_patch_height = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            resized_patch_width=resized_patch_width,
            resized_patch_height=resized_patch_height
        )

class adapt_CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = adapt_CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        resized_patch_width = None, 
        resized_patch_height = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, resized_patch_width, resized_patch_height)
        hidden_states = self.pre_layrnorm(hidden_states)

        sums = hidden_states.sum(dim=-1)
        attention_mask = (sums==-1.0000).float()
        attention_mask[attention_mask==1] = -float('inf')

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1,1,577,1).to(hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class adapt_CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def get_adapt_position_embedding(self, position_embedding, patch_width_num, patch_height_num):
        position_embedding = position_embedding.squeeze(0)
        position_for_class = position_embedding[0:1, :]


        position_embedding = position_embedding[1:, :].reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, POSITION_EMBEDDING_LENGTH)
        position_embedding = position_embedding.permute(2,0,1).unsqueeze(0) #[1, d, h, w]
        original_dtype = position_embedding.dtype

        position_embedding = position_embedding.to(torch.float)
        position_embedding = F.interpolate(position_embedding, size=(patch_height_num, patch_width_num), mode='bilinear', align_corners=False)

        position_embedding = position_embedding.to(original_dtype)
        position_embedding = position_embedding.squeeze(0).permute(1,2,0).reshape(patch_height_num*patch_width_num, POSITION_EMBEDDING_LENGTH)

        # zero padding
        position_embedding = F.pad(position_embedding, (0,0,0,MAX_PATCHES-patch_height_num*patch_width_num))
        position_embedding = torch.cat((position_for_class, position_embedding), dim=0)
        return position_embedding


    def forward(self, 
        pixel_values: torch.FloatTensor,
        resized_patch_width, 
        resized_patch_height
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        adapt_position_embedding = torch.cat([
            self.get_adapt_position_embedding(
                self.position_embedding(self.position_ids),
                patch_width_num = d[0],
                patch_height_num = d[1],
            ).unsqueeze(0) for d in list(zip(resized_patch_width, resized_patch_height))
        ])

        embeddings = embeddings + adapt_position_embedding
        return embeddings