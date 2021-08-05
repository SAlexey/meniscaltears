# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import math
from models.resnet import resnet18_3d, resnet34_3d, resnet50_3d

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Union

from util.misc import NestedTensor
from hydra.utils import call


BACKBONE = {
    "resnet18": (torchvision.models.resnet18, 512),
    "resnet34": (torchvision.models.resnet34, 512),
    "resnet50": (torchvision.models.resnet50, 2048),
    "resnet18_3d": (resnet18_3d, 512),
    "resnet34_3d": (resnet34_3d, 512),
    "resnet50_3d": (resnet50_3d, 2048),
}


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self, backbone: nn.Module, num_channels: int, dim: Union[int, float] = 3
    ):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels
        self.dim = float(dim)
        assert self.dim in (
            2.0,
            2.5,
            3.0,
        ), f"dim ({dim}) must be one of (2.0, 2.5, 3.0)"

    def forward(self, tensor_list: NestedTensor, attention=True):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():

            if attention:
                m = tensor_list.mask
                assert m is not None
                size = x.shape[-int(math.floor(self.dim)) :]
                mask = F.interpolate(m[None].float(), size=size).to(torch.bool)[0]
            else:
                mask = None
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor, attention: bool = True):
        features, position = self
        xs = features(tensor_list, attention)
        out: List[NestedTensor] = []
        pos: List[torch.Tensor] = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            if attention:
                pos.append(position(x).to(x.tensors.dtype))
        return out, pos
