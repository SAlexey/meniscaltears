# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import einops
from util.misc import NestedTensor
import numpy as np


class PositionEmbeddingSine1d(nn.Module):
    def __init__(
        self, hidden_dim, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor, dim: int = -1):

        x = tensor_list.tensors
        emb = torch.ones(x.size(dim))
        emb = emb.cumsum(dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            emb = emb / (emb + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = emb[None] / dim_t
        pos = torch.stack((pos[:, 0::2].sin(), pos[:, 1::2].cos()), dim=-1).flatten(-2)

        
        return pos


class PositionEmbeddingSine2d(PositionEmbeddingSine1d):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, hidden_dim, *args, **kwargs):
        assert hidden_dim % 2 == 0
        super().__init__(hidden_dim, *args, **kwargs)
        self.num_pos_feats = hidden_dim // 2

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(-2, dtype=torch.float32)
        x_embed = not_mask.cumsum(-1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingSine2p1d(PositionEmbeddingSine2d):
    def forward(self, tensor_list: NestedTensor):
        pos = super().forward(tensor_list)
        mask = tensor_list.mask
        not_mask = ~mask

        # batch position
        z_embed = not_mask.cumsum(0, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :] + eps) * self.scale


class PositionEmbeddingSine3d(PositionEmbeddingSine2d):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, hidden_dim, *args, **kwargs):
        assert hidden_dim % 3 == 0
        super().__init__(hidden_dim, *args, **kwargs)
        self.num_pos_feats = hidden_dim // 3

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:  # allow the mask to be none since all images are the same size
            mask = torch.zeros_like(x)
        not_mask = ~mask  # mask shape [N d h w]

        z_embed = not_mask.cumsum(-3, dtype=torch.float32)
        y_embed = not_mask.cumsum(-2, dtype=torch.float32)
        x_embed = not_mask.cumsum(-1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_z = z_embed[..., None] / dim_t
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_z = torch.stack(
            (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=-1).permute(0, 4, 1, 2, 3)
        return pos


def build(args, hidden_dim):
    N_steps = hidden_dim // 2
    if args.position_embedding == "v2":
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine2d(N_steps, normalize=True)
    elif args.position_embedding == "v3":
        position_embedding = PositionEmbeddingSine3d(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
