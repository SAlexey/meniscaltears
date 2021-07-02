#%%
from util.misc import _is_numeric, _is_sequence
from util.box_ops import box_xyxy_to_cxcywh, normalize_boxes
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from numbers import Number
from random import randint, random


class Compose(T.Compose):
    def __call__(self, img, tgt):
        for t in self.transforms:
            img, tgt = t(img, tgt)
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

    img = F.interpolate(img.unsqueeze(0), size).squeeze(0)

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


def random_bbox_safe_crop(img, tgt):
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

    mins = torch.min(tgt["boxes"], 0).values[:3]
    maxs = torch.max(tgt["boxes"], 0).values[-3:]

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
                    if _is_numeric(v) or (_is_sequence(v) and _is_numeric(v)):
                        tgt[k] = torch.as_tensor(v)
            elif _is_numeric(tgt) or (_is_sequence(tgt) and _is_numeric(tgt)):
                tgt = torch.as_tensor(tgt)
        return img, tgt


class Normalize(object):
    def __call__(self, img, tgt=None, mean=(0.485,), std=(0.229,)):

        img = img / img.max()
        img = (img - torch.as_tensor(mean)) / torch.as_tensor(std)

        if tgt is not None:

            tgt = tgt.copy()

            if "boxes" in tgt:
                boxes = tgt["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = normalize_boxes(boxes, size=tuple(img.size()[-3:]))
                tgt["boxes"] = boxes

        return img, tgt


class RandomResizedBBoxSafeCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tgt):

        tgt = tgt.copy()
        size = img.size()[-3:]

        if random() <= self.p:
            crop = random_bbox_safe_crop(img, tgt)

            img, tgt = crop_volume(img, crop, tgt=tgt)
            img, tgt = resize_volume(img, size, tgt=tgt)

        return img, tgt
