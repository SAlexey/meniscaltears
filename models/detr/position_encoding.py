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


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:  # allow the mask to be none since all images are the same size
            mask = torch.zeros_like(x)
        not_mask = ~mask  # mask shape [N d h w]

        z_embed = not_mask.cumsum(-3, dtype=torch.float32)
        y_embed = not_mask.cumsum(-2, dtype=torch.float32)
        x_embed = not_mask.cumsum(-1, dtype=torch.float32)

        print(z_embed.shape)
        print(z_embed[:, -1, :, :].shape)
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


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


class PositionEmbeddingLearned3d(nn.Module):
    def __init__(self, num_pos_feats=256, num_queries=6):
        super().__init__()
        assert num_queries % 3 == 0, "Num queries must be divisible by 3!"
        steps = num_queries // 3
        self.x_emb = nn.Embedding(steps, num_pos_feats)
        self.y_emb = nn.Embedding(steps, num_pos_feats)
        self.z_emb = nn.Embedding(steps, num_pos_feats)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_emb.weight)
        nn.init.uniform_(self.y_emb.weight)
        nn.init.uniform_(self.z_emb.weight)

    def forward(self, tensor_list: NestedTensor):

        x = tensor_list.tensors
        bs, _, h, w, d = x.shape
        i = torch.arange(h, device=x.device)
        j = torch.arange(w, device=x.device)
        k = torch.arange(d, device=x.device)

        x_emb = einops.rearrange(self.x_emb(i), "x -> 1 1 x")
        y_emb = einops.rearrange(self.y_emb(j), "y -> 1 y 1")
        z_emb = einops.rearrange(self.z_emb(k), "z -> z 1 1")

        x_emb = einops.repeat(x_emb, "x y z -> (x h) y z", h=h)
        y_emb = einops.repeat(y_emb, "x y z -> x (y w) z", w=w)
        z_emb = einops.repeat(z_emb, "x y z -> x y (z d)", d=d)

        pos = einops.rearrange([x_emb, y_emb, z_emb], "list x y z ->1 list x y z")
        pos = einops.repeat(pos, "batch c x y z -> (batch size) c x y z", size=bs)

        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding == "learned3d":
        position_embedding = PositionEmbeddingLearned3d(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
