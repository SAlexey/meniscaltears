from functools import partial

from torch.functional import norm
from einops.einops import rearrange
import torch
from torch import nn
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import einops as E
from einops.layers.torch import Rearrange
from .linear import MLP
from omegaconf import DictConfig
import os
import copy


def conv3x3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    padding=None,
) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation if not padding else padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def stem(name):
    CHOICES = {
        "basic_res": BasicStem3D,
        "basic_377": BasicStem377, 
        "basic": models.video.resnet.BasicStem,
        "2plus1d": models.video.resnet.R2Plus1dStem,
    }
    return CHOICES[name]


def block(name):
    CHOICES = {
        "basic_res": BasicBlock3D,
        "basic_vid": models.video.resnet.BasicBlock,
        "bottleneck_res": Bottleneck3D,
        "bottleneck_vid": models.video.resnet.Bottleneck,
    }
    return CHOICES[name]


def conv(name):
    CHOICES = {
        "2plus1d": models.video.resnet.Conv2Plus1D,
        "3d_simple": models.video.resnet.Conv3DSimple,
        "3d_notemp": models.video.resnet.Conv3DNoTemporal,
    }
    return CHOICES[name]


class BasicStem3D(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7, 7),
        stride=(2, 2, 2),
        padding=(3, 3, 3),
        bias=False,
    ):
        super(BasicStem3D, self).__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicStem377(BasicStem3D):

    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        kernel_size=(3, 7, 7),
        stride=(2, 2, 2),
        padding=(3, 3, 3),
        bias=False,
    ):

        super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(
        self,
        in_channels=1,
        kernel_size=(7, 3),
        padding=(3, 1),
        stride=(2, 1),
        bias=False,
    ):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2

        k0, k1 = kernel_size
        k0 = (1,) + (k0,) * 2
        k1 = (k1,) + (1,) * 2

        if isinstance(padding, int):
            padding = (padding,) * 2

        p0, p1 = padding
        p0 = (0,) + (p0,) * 2
        p1 = (p1,) + (0,) * 2

        if isinstance(stride, int):
            stride = (stride,) * 2

        s0, s1 = stride

        s0 = (1,) + (s0,) * 2
        s1 = (s1,) + (1,) * 2

        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(
                in_channels,
                45,
                kernel_size=k0,
                stride=s0,
                padding=p0,
                bias=bias,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=k1,
                stride=s1,
                padding=p1,
                bias=bias,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class BasicBlock3D(BasicBlock):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


class Bottleneck3D(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DilationBasic3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation=(1, 1),
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        residual=True,
    ) -> None:
        super(DilationBasic3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(
            inplanes, planes, stride, padding=dilation[0], dilation=dilation[0]
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3x3(
            planes, planes, padding=dilation[1], dilation=dilation[1]
        )
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual = residual
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.residual:
            out += identity
        out = self.relu(out)

        return out


def resnet18_3d(*, block=BasicBlock3D, norm_layer=nn.BatchNorm3d, **kwargs) -> ResNet:
    return ResNet3D(block, [2, 2, 2, 2], norm_layer=norm_layer, **kwargs)


def resnet34_3d(*, block=BasicBlock3D, norm_layer=nn.BatchNorm3d, **kwargs) -> ResNet:
    return ResNet3D(block, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)


def resnet50_3d(*, block=Bottleneck3D, norm_layer=nn.BatchNorm3d, **kwargs) -> ResNet:
    stem = BasicStem3D(3)
    return ResNet3D(stem, block, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)


def resnet50_3d_v2(*args, **kwargs):
    model = resnet50_3d(*args, **kwargs)
    model.layer4 = nn.Sequential(
        nn.Conv3d(1024, 1024, 1),
        nn.BatchNorm3d(1024),
        nn.ReLU(True),
        nn.Conv3d(1024, 2048, 1),
        nn.BatchNorm3d(2048),
        nn.ReLU(True),
        nn.Conv3d(2048, 2048, 1),
        nn.BatchNorm3d(2048),
        nn.ReLU(True),
    )
    return model


def wide_resnet50_3d(
    *, block=Bottleneck3D, norm_layer=nn.BatchNorm3d, width_per_group=64 * 2, **kwargs
) -> ResNet:

    stem = BasicStem3D(3)
    return ResNet3D(
        stem,
        block,
        [3, 4, 6, 3],
        norm_layer=norm_layer,
        width_per_group=width_per_group,
        **kwargs,
    )


def wide_resnet50_3d_v2(*args, **kwargs):
    model = wide_resnet50_3d(*args, **kwargs)
    model.layer4 = nn.Sequential(
        nn.Conv3d(1024, 1024, 1),
        nn.BatchNorm3d(1024),
        nn.ReLU(True),
        nn.Conv3d(1024, 2048, 1),
        nn.BatchNorm3d(2048),
        nn.ReLU(True),
    )
    return model


def dilated_resnet26_3d(
    *, block=DilationBasic3D, norm_layer=nn.BatchNorm3d, **kwargs
) -> ResNet:
    return DilationResNet3D(
        block, [1, 1, 2, 2, 2, 2, 1, 1], norm_layer=norm_layer, **kwargs
    )


class ResNet3D(nn.Module):
    def __init__(
        self,
        stem: nn.Module,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
        pretrained=False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.stem = stem()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: BasicBlock3D,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x -> (BS, 1, 160, 320, 320)
        x = self.stem(x)
        x = self.maxpool(x)  # x -> (BS, 64, 40, 80, 80)

        x = self.layer1(x)  # x -> (BS, 128, 20, 40, 40)
        x = self.layer2(x)  # x -> (BS, 256, 10, 20, 20)
        x = self.layer3(x)  # x -> (BS, )
        x = self.layer4(x)  # x -> (BS, 512, 5, 10, 10)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class RNet(ResNet3D):
    def __init__(self, block, *args, **kwargs):
        super().__init__(block, *args, **kwargs)
        self.layer4 = nn.Sequential(
            nn.Conv3d(
                self.inplanes // block.expansion // 2,
                self.inplanes // block.expansion,
                kernel_size=1,
            ),
            nn.BatchNorm3d(self.inplanes // block.expansion),
            nn.ReLU(True),
            nn.Conv3d(
                self.inplanes // block.expansion,
                self.inplanes // block.expansion,
                kernel_size=1,
            ),
            nn.BatchNorm3d(self.inplanes // block.expansion),
            nn.ReLU(True),
            nn.Conv3d(self.inplanes // block.expansion, self.inplanes, kernel_size=1),
            nn.BatchNorm3d(self.inplanes),
            nn.ReLU(True),
        )


class DilationResNet3D(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        channels=(16, 32, 64, 128, 256, 512, 512, 512),
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.inplanes = channels[0]

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.stem = BasicStem3D()

        self.layer0 = self._make_layer(
            block=block, planes=channels[0], blocks=layers[0], stride=1
        )
        self.layer1 = self._make_layer(
            block=block, planes=channels[1], blocks=layers[1], stride=2
        )
        self.layer2 = self._make_layer(
            block=block, planes=channels[2], blocks=layers[2], stride=2
        )
        self.layer3 = self._make_layer(
            block=block, planes=channels[3], blocks=layers[3], stride=2
        )
        self.layer4 = self._make_layer(
            block=block,
            planes=channels[4],
            blocks=layers[4],
            dilation=2,
            new_level=False,
        )
        self.layer5 = self._make_layer(
            block=block,
            planes=channels[5],
            dilation=4,
            blocks=layers[5],
            new_level=False,
        )
        self.layer6 = self._make_layer(
            block=block,
            planes=channels[6],
            dilation=2,
            blocks=layers[6],
            new_level=False,
            residual=False,
        )
        self.layer7 = self._make_layer(
            block=block,
            planes=channels[7],
            dilation=1,
            blocks=layers[7],
            residual=False,
            new_level=False,
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: BasicBlock3D,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        new_level: bool = True,
        residual: bool = True,
    ) -> nn.Sequential:
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        norm_layer = self._norm_layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=(1, 1)
                if dilation == 1
                else (dilation // 2 if new_level else dilation, dilation),
                downsample=downsample,
                norm_layer=norm_layer,
                residual=residual,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=(dilation, dilation),
                    norm_layer=norm_layer,
                    residual=residual,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x -> (BS, 1, 160, 300, 300)
        x = self.stem(x)

        x = self.layer0(x)  # x -> (BS, 128, 20, 38, 38)
        x = self.layer1(x)  # x -> (BS, 128, 20, 38, 38)
        x = self.layer2(x)  # x -> (BS, 256, 10, 19, 19)
        x = self.layer3(x)  # x -> (BS, )
        x = self.layer4(x)  # x -> (BS, 512, 5, 10, 10)
        x = self.layer5(x)  # x -> (BS, 128, 20, 38, 38)
        x = self.layer6(x)  # x -> (BS, 256, 10, 19, 19)
        x = self.layer7(x)  # x -> (BS, )

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _grayscale_resnet(name, *args, **kwargs):
    resnet = getattr(models, name)(*args, **kwargs)
    old_conv1 = resnet.conv1

    new_conv1 = nn.Conv2d(
        in_channels=old_conv1.in_channels,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=old_conv1.bias is not None,
    )

    del resnet.fc

    new_conv1.weight.data = old_conv1.weight.data.mean(1, keepdim=True)
    resnet.conv1 = new_conv1
    return resnet


CONFIG = {
    "resnet18": (partial(_grayscale_resnet, "resnet18"), 512),
    "resnet34": (partial(_grayscale_resnet, "resnet34"), 512),
    "resnet50": (partial(_grayscale_resnet, "resnet50"), 2048),
    "resnet18_3d": (resnet18_3d, 512),
    "resnet34_3d": (resnet34_3d, 512),
    "resnet50_3d": (resnet50_3d, 2048),
    "dilated_resnet26_3d": (dilated_resnet26_3d, 512),
}


class Net1(nn.Module):
    """
    Detects 2 Binary Labels:
        - medial meniscus [healthy / diseased]
        - lateral meniscus [healthy / diseased]
    """

    def __init__(
        self,
        backbone: str,
        *args,
        hidden_dim=1024,
        output_dim=6,
        num_layers=1,
        **kwargs,
    ):
        """
        Args:
            backbone: (string)
                which backbone  to use. Choices are (resnet18_3d, resnet50_3d)
            hidden_dim: (int)
                dimmension of the MLP
            output_dim: (int)
                dimension of the output logits of MLP
            num_layers: (int)
                number of layers in the MLP
        """

        assert backbone in CONFIG, f"unknown backbone: {backbone}"
        super().__init__()
        init, num_channels = CONFIG[backbone]
        self.backbone = init(*args, **kwargs)
        self.out_labels = MLP(
            input_dim=num_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.out_labels(x)
        return {"labels": x}


class Net2(Net1):
    """
    Detects menisci and classifies them in a multilabel & multiclass manner (MOAKS)

    Outputs: a dictionary with keys
        - boxes: tensor[12] box coordinates in cxcywh format
    """

    def __init__(
        self,
        backbone: str,
        *args,
        hidden_dim=1024,
        output_dim=6,
        num_layers=1,
        det_hidden_dim=1024,
        det_output_dim=12,
        det_num_layers=3,
        **kwargs,
    ):
        assert backbone in CONFIG, f"unknown backbone: {backbone}"
        super().__init__(
            backbone,
            *args,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            **kwargs,
        )

        self.out_boxes = MLP(
            input_dim=CONFIG[backbone][1],
            hidden_dim=det_hidden_dim,
            output_dim=det_output_dim,
            num_layers=det_num_layers,
        )

    def forward(self, x):
        x = self.backbone(x)
        out = {"labels": self.out_labels(x), "boxes": self.out_boxes(x)}
        return out


class ClsNet2D(nn.Module):
    """
    Classify menisci using pretrained 2d resnet

    expected input:
     Grayscale 3d Image: torch.Tensor [1, 1, D, H, W]

    output:
     dictionary {
         labels: torch.Tensor [1, NC] - flattened labels
     }

     IMPORTANT: ONLY BATCH SIZE OF 1 IS SUPPORTED!
    """

    def __init__(
        self,
        backbone,
        *args,
        dropout=0.0,
        cls_output_dim=6,
        cls_hidden_dim=2048,
        cls_dropout=0.0,
        cls_num_layers=1,
        **kwargs,
    ):
        super().__init__()
        assert backbone in CONFIG, f"unknown backbone: {backbone}"
        backbone, num_channels = CONFIG[backbone]
        backbone = backbone(*args, **kwargs)
        self.backbone = IntermediateLayerGetter(backbone, {"layer4": "features"})
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.batch_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout)

        self.out_cls = MLP(
            input_dim=num_channels,
            hidden_dim=cls_hidden_dim,
            output_dim=cls_output_dim,
            num_layers=cls_num_layers,
            dropout=cls_dropout,
        )

    def forward(self, x, return_features=False):
        x = rearrange(x, "bs ch d h w -> (bs d) ch h w")
        x = self.backbone(x)["features"]
        x = self.feature_pool(x)
        x = rearrange(x, "bs c h w  -> c (h w) bs")
        x = self.batch_pool(x)
        x = rearrange(x, "c hw bs -> bs (c hw)")
        x = self.dropout(x)
        out = {"labels": self.out_cls(x)}
        if return_features:
            return x, out
        return out


class DetNet2D(ClsNet2D):
    """
    Detect and classify menisci using pretrained 2d resnet

    expected input:
     Grayscale 3d Image: torch.Tensor [1, 1, D, H, W]

    output:
     dictionary {
         boxes: torch.Tensor [1, 12] - flattened box coords (2 boxes 6 coords each)
         labels: torch.Tensor [1, NC] - flattened labels
     }

     IMPORTANT: ONLY BATCH SIZE OF 1 IS SUPPORTED!
    """

    def __init__(
        self,
        backbone,
        *args,
        det_output_dim=12,
        det_hidden_dim=2048,
        det_num_layers=2,
        det_dropout=0.0,
        **kwargs,
    ):
        super().__init__(backbone, *args, **kwargs)
        _, num_channels = CONFIG[backbone]
        self.out_box = MLP(
            input_dim=num_channels,
            hidden_dim=det_hidden_dim,
            output_dim=det_output_dim,
            num_layers=det_num_layers,
            dropout=det_dropout,
        )

    def forward(self, x, return_features=False):
        x, out = super().forward(x, return_features=True)
        out["boxes"] = self.out_box(x)
        if return_features:
            return x, out
        return out


class ClsNet3D(nn.Module):
    def __init__(
        self,
        backbone,
        *args,
        dropout=0.0,
        cls_output_dim=6,
        cls_hidden_dim=2048,
        cls_num_layers=1,
        cls_dropout=0.0,
        intermediate_layer="layer4",
        **kwargs,
    ):
        super().__init__()

        backbone, num_channels = CONFIG[backbone]
        backbone = backbone(*args, **kwargs)
        self.intermediate_layer = intermediate_layer
        self.backbone = IntermediateLayerGetter(
            backbone, {intermediate_layer: "features"}
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.out_cls = MLP(
            input_dim=num_channels,
            hidden_dim=cls_hidden_dim,
            output_dim=cls_output_dim,
            num_layers=cls_num_layers,
            dropout=cls_dropout,
        )

    def forward(self, x, return_features=False):
        x = self.backbone(x)["features"]
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        out = {"labels": self.out_cls(x)}

        if return_features:
            return x, out

        return out


class DetNet3D(ClsNet3D):
    def __init__(
        self,
        backbone,
        *args,
        det_hidden_dim=2048,
        det_output_dim=12,
        det_num_layers=2,
        det_dropout=0.0,
        **kwargs,
    ):
        super().__init__(backbone, *args, **kwargs)
        _, num_channels = CONFIG[backbone]
        self.out_box = MLP(
            input_dim=num_channels,
            hidden_dim=det_hidden_dim,
            output_dim=det_output_dim,
            num_layers=det_num_layers,
            dropout=det_dropout,
        )

    def forward(self, x):
        x, out = super().forward(x, return_features=True)
        out["boxes"] = self.out_box(x)
        return out
