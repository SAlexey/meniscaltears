#%%
from util.misc import _is_numeric, _is_sequence
from util.box_ops import box_xyxy_to_cxcywh, normalize_boxes
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from numbers import Number
from random import randint, random
import math
import numpy as np
from skimage.util import invert
from torchvision.transforms import functional as TF
from scipy import ndimage


class Compose(T.Compose):
    def __call__(self, img, target=None, size=None):
        for t in self.transforms:
            if isinstance(t, CropIMG):
                img, target = t(img, target, size)

            else:
                img, target = t(img, target)
        return img, target


class NoOp(object):
    def __call__(self, img, tgt=None, **kwargs):
        return img, tgt


def _apply_crop_to_boxes(boxes, crop):

    zmin, ymin, xmin, zmax, ymax, xmax = boxes.unbind(-1)

    d = zmax - zmin
    h = ymax - ymin
    w = xmax - xmin

    zmin = zmin - crop[0]
    ymin = ymin - crop[1]
    xmin = xmin - crop[2]

    zmax = zmin + d
    ymax = ymin + h
    xmax = xmin + w

    return torch.stack((zmin, ymin, xmin, zmax, ymax, xmax), -1)


def _apply_resize_to_boxes(boxes, scale):

    zmin, ymin, xmin, zmax, ymax, xmax = boxes.unbind(-1)

    cz = (zmax + zmin) * 0.5 * scale[0]
    cy = (ymax + ymin) * 0.5 * scale[1]
    cx = (xmax + xmin) * 0.5 * scale[2]

    d = (zmax - zmin) * scale[0]
    h = (ymax - ymin) * scale[1]
    w = (xmax - xmin) * scale[2]

    zmin = cz - (d * 0.5)
    ymin = cy - (h * 0.5)
    xmin = cx - (w * 0.5)

    zmax = cz + (d * 0.5)
    ymax = cy + (h * 0.5)
    xmax = cx + (w * 0.5)

    return torch.stack((zmin, ymin, xmin, zmax, ymax, xmax), -1)


def _assert_tgt(tgt):
    assert isinstance(tgt, dict) and ("boxes" in tgt)
    boxes = tgt["boxes"]
    assert (
        isinstance(boxes, torch.Tensor) and (boxes.ndim == 2) and (boxes.size(-1) == 6)
    )


def _assert_img(img, crop=None, size=None):
    assert img.ndim == 4

    if crop is not None:
        assert len(crop) == 6

    if size is not None:
        assert len(size) == 3


def resize_volume(img, size, tgt=None):
    """
    resize image, adjust bounding boxes
    """
    _assert_img(img, size=size)

    sx, sy, sz = size
    ox, oy, oz = img.size()[-3:]

    zoom = (1, sx / ox, sy / oy, sz / oz)

    img = F.interpolate(
        img.unsqueeze(0), size, mode="trilinear", align_corners=False
    ).squeeze(0)

    if tgt is not None:
        _assert_tgt(tgt)
        tgt = tgt.copy()
        tgt["boxes"] = _apply_resize_to_boxes(tgt["boxes"], zoom[1:])

    return img, tgt


def crop_volume(img, crop, tgt=None):
    """
    crop image, adjust bounding boxes
    """

    _assert_img(img, crop=crop)

    back, top, left, depth, height, width = crop
    img = img[..., back : back + depth, top : top + height, left : left + width]

    if tgt is not None:
        _assert_tgt(tgt)
        tgt = tgt.copy()
        tgt["boxes"] = _apply_crop_to_boxes(tgt["boxes"], crop)

    return img, tgt


def random_bbox_safe_crop(img, tgt, bbox_safe=False):
    """
    random crop that preserves the bounding box

    Args:
        img (Tensor[..., D, H, W])
        tgt (dict[boxes!,...])
    Return
        crop (tuple): (back, top, left, depth, height, width)

    Notes:
         0 <= back <= min_z(boxes)
         0 <= top <= min_y(boxes)
         0 <= left <= min_x(boxes)

         max_z(boxes) <= depth <= img_depth
         max_y(boxes) <= height <= img_height
         max_x(boxes) <= width <= img_width
    """

    if bbox_safe:
        mins = torch.min(tgt["boxes"], 0).values[:3]
        maxs = torch.max(tgt["boxes"], 0).values[-3:]

    else:

        d, h, w = img.size()[-3:]

        # preserve 95% of the depth
        d = int(round(d * 0.975))

        # preserve 80% of the height
        h = int(round(h * 0.90))

        # preserve 80% of the width
        w = int(round(w * 0.90))

        mins = torch.as_tensor((img.size(-3) - d, img.size(-2) - h, img.size(-2) - w))
        maxs = torch.as_tensor((d, h, w))

    zmin = randint(0, mins[0])
    ymin = randint(0, mins[1])
    xmin = randint(0, mins[2])

    zmax = randint(maxs[0], img.size(1))
    ymax = randint(maxs[1], img.size(2))
    xmax = randint(maxs[2], img.size(3))

    depth = zmax - zmin
    height = ymax - ymin
    width = xmax - xmin

    return (zmin, ymin, xmin, depth, height, width)


class ToTensor(object):
    def __call__(self, img, tgt=None):
        img = torch.as_tensor(img).float()
        if tgt is not None:
            if isinstance(tgt, dict):
                tgt = tgt.copy()
                for k, v in tgt.items():
                    tgt[k] = torch.as_tensor(v)
        return img, tgt


class Normalize(object):
    def __init__(self, mean=(0,), std=(1,)):

        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def __call__(self, img, tgt=None):

        img = img / img.max()
        img = (img - self.mean) / self.std

        if tgt is not None:

            tgt = tgt.copy()

            if "boxes" in tgt:
                boxes = tgt["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = normalize_boxes(boxes, size=tuple(img.size()[-3:]))
                tgt["boxes"] = boxes

        return img, tgt


class CenterCropVolume(object):
    def __init__(self, output_size=(160, 360, 360)):
        self.size = output_size

    def __call__(self, img, tgt=None):

        id, ih, iw = img.size()[-3:]
        od, oh, ow = self.size

        back, top, left = 0, 0, 0
        depth, height, width = id, ih, iw

        dd, dh, dw = id - od, oh - ih, ow - iw

        if dd > 0:  # only crop if the output has to be smaller
            back = math.ceil(dd / 2)
            depth = id - back

        if dh > 0:
            top = math.ceil(dh / 2)
            height = ih - top

        if dw > 0:
            left = math.ceil(dw / 2)
            width = iw - left

        crop = (back, top, left, depth, height, width)

        return crop_volume(img, crop, tgt)


class TopHatFilter(object):
    def __init__(self, b_size, w_size):
        self.bs = b_size
        self.ws = w_size

    def __call__(self, img, tgt=None):
        bth = ndimage.black_tophat(img, size=self.bs)
        wth = ndimage.white_tophat(img, size=self.ws)

        bth = torch.from_numpy(bth)
        wth = torch.from_numpy(wth)

        return torch.cat((img, bth, wth)), tgt


class Resize(object):
    def __init__(self, size=(160, 384, 384)):
        self.size = size

    def __call__(self, input, tgt=None):
        return resize_volume(input, self.size, tgt=tgt)


class RandomInvert(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tgt=None):
        if random() <= self.p:
            img = invert(img.numpy())
            img = torch.from_numpy(img)
        return img, tgt


class AugSmoothTransform(object):
    def __init__(self, p=0.5):
        self.p = p
        self.noise = torch.distributions.Normal(0.0, 0.2)
        self.transforms = [
            self.random_hflip,
            self.random_noise,
            self.random_rotate,
            self.random_multiply,
        ]

    def random_hflip(self, img):
        if random() < self.p:
            return img.flip(-1)
        return img

    def random_noise(self, img):
        if random() <= self.p:
            return img + self.noise.sample(img.size()).to(img.device)
        return img

    def random_rotate(self, img):
        if random() < self.p:
            angle = np.random.uniform(-5, +5)
            rotated_img = [TF.rotate(i, angle) for i in img.unbind(2)]
            return torch.stack(rotated_img, dim=2)
        return img

    def random_multiply(self, img):
        if random() <= self.p:
            return img * np.random.uniform(0.9, 1.1)
        return img

    def __call__(self, img, tgt=None):
        for transform in self.transforms:
            img = transform(img)
        return img, tgt


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor, tgt=None):
        if random() <= self.p:
            tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor, tgt

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomResizedBBoxSafeCrop(object):
    def __init__(self, p=0.5, bbox_safe=False):
        self.p = p
        self.safe = bbox_safe

    def __call__(self, img, tgt):

        tgt = tgt.copy()
        size = img.size()[-3:]

        if random() <= self.p:

            crop = random_bbox_safe_crop(img, tgt, bbox_safe=self.safe)

            img, tgt = crop_volume(img, crop, tgt=tgt)
            img, tgt = resize_volume(img, size, tgt=tgt)

        return img, tgt


class CropIMG(object):
    def __init__(self, p=0.7, random=True):
        self.p = p
        self.random = random

    def get_crop(self, img, target, size_faktor=0.2):
        """
        random crop that preserves the bounding box

        Args:
            img (Tensor[..., D, H, W])
            tgt (dict[boxes!,...])
        Return
            crop (tuple): (back, top, left, depth, height, width)

        Notes:
            0 <= back <= min_z(boxes)
            0 <= top <= min_y(boxes)
            0 <= left <= min_x(boxes)

            max_z(boxes) <= depth <= img_depth
            max_y(boxes) <= height <= img_height
            max_x(boxes) <= width <= img_width
        """

        mins = torch.min(target["boxes"], 0).values[:3]
        maxs = torch.max(target["boxes"], 0).values[-3:]

        if self.random and random() <= self.p:
            zmin = randint(
                max(0, int(math.floor(mins[0] * (1 - size_faktor)))), mins[0]
            )
            ymin = randint(
                max(0, int(math.floor(mins[1] * (1 - size_faktor)))), mins[1]
            )
            xmin = randint(
                max(0, int(math.floor(mins[2] * (1 - size_faktor)))), mins[2]
            )

            zmax = randint(
                maxs[0], int(math.ceil(min(img.size(1), maxs[0] * (1 + size_faktor))))
            )
            ymax = randint(
                maxs[1], int(math.ceil(min(img.size(2), maxs[1] * (1 + size_faktor))))
            )
            xmax = randint(
                maxs[2], int(math.ceil(min(img.size(3), maxs[2] * (1 + size_faktor))))
            )
        else:
            size_faktor = 0.1
            zmin = max(0, int(math.floor(mins[0] * (1 - size_faktor / 2))))
            ymin = max(0, int(math.floor(mins[1] * (1 - size_faktor / 2))))
            xmin = max(0, int(math.floor(mins[2] * (1 - size_faktor / 2))))

            zmax = min(img.size(1), int(math.ceil(maxs[0] * (1 + size_faktor / 2))))
            ymax = min(img.size(2), int(math.ceil(maxs[1] * (1 + size_faktor / 2))))
            xmax = min(img.size(3), int(math.ceil(maxs[2] * (1 + size_faktor / 2))))

        depth = zmax - zmin
        height = ymax - ymin
        width = xmax - xmin

        return (zmin, ymin, xmin, depth, height, width)

    def resize_volume(self, img, size):
        """
        resize image
        """
        _assert_img(img, size=size)
        img = F.interpolate(
            img.unsqueeze(0), size, mode="trilinear", align_corners=False
        ).squeeze(0)
        return img

    def crop_volume(self, img, crop):
        """
        crop image
        """
        _assert_img(img, crop=crop)
        back, top, left, depth, height, width = crop
        return img[..., back : back + depth, top : top + height, left : left + width]

    def __call__(self, img, target, size):
        crop = self.get_crop(img, target)
        img = self.crop_volume(img, crop)
        img = self.resize_volume(img, size)
        return img, target
