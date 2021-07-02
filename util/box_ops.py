import numpy as np
import torch
from scipy import ndimage


def box_xyxy_to_cxcywh(boxes):
    assert (boxes[:, 3:] >= boxes[:, :3]).all(), "corrupt boxes or not xyxy"

    x0, y0, z0, x1, y1, z1 = boxes.unbind(-1)

    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    cz = (z0 + z1) * 0.5

    w, h, d = x1 - x0, y1 - y0, z1 - z0

    box = torch.stack((cx, cy, cz, w, h, d), -1)
    return box


def denormalize_boxes(boxes: torch.Tensor, size=(160, 384, 384)):
    return (boxes * torch.as_tensor(size + size)).round().int()


def normalize_boxes(boxes: torch.Tensor, size=(160, 384, 384)):
    return boxes / torch.as_tensor(size + size)


def box_cxcywh_to_xyxy(boxes):

    cx, cy, cz, w, h, d = boxes.unbind(-1)

    x0 = cx - w * 0.5
    x1 = cx + w * 0.5

    y0 = cy - h * 0.5
    y1 = cy + h * 0.5

    z0 = cz - d * 0.5
    z1 = cz + d * 0.5

    box = torch.stack((x0, y0, z0, x1, y1, z1), -1)

    return box


def _objects_to_boxes(objects):
    boxes = []
    for obj in objects:
        mins = []
        maxs = []
        if obj is not None:
            for ax in obj:
                mins.append(ax.start)
                maxs.append(ax.stop)
        boxes.append(mins + maxs)
    return boxes


def find_objects(mask, boxes=True, **kwargs):
    objects = ndimage.find_objects(mask, **kwargs)
    if not boxes:
        return objects
    return _objects_to_boxes(objects)


def box_volume(boxes):
    """
    Computes the volume of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    boxes = _upcast(boxes)
    return (
        (boxes[:, 3] - boxes[:, 0])
        * (boxes[:, 4] - boxes[:, 1])
        * (boxes[:, 5] - boxes[:, 2])
    )


def _upcast(t):
    """COPY PASTE FROM TORCHVISION OPS"""
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def _box_inter_union(boxes1, boxes2):
    """
    COPY PASTE FROM TORCHVISION OPS
    CHANGED TO WORK IN 3D
    """

    vol1 = box_volume(boxes1)
    vol2 = box_volume(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,3]
    inter = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]  # [N,M]

    union = vol1[:, None] + vol2 - inter

    return inter, union


def box_iou(boxes1, boxes2):
    """
    COPY PASTE FROM TORCHVISION OPS
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """

    COPY PASTE FROM TORCHVISION OPS
    works in 3d

    Return generalized intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        generalized_iou (Tensor[N, M]): the NxM matrix containing the pairwise generalized_IoU values
        for every element in boxes1 and boxes2
    """

    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union

    lti = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rbi = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1] * whi[:, :, 2]

    return iou - (areai - union) / areai


def box_img(boxes, size=(160, 384, 384)):
    """
    Create a 3d image with filled boxes

    Args:
        boxes (Tensor[N, 6]): bboxes in [x1 y1 z1 x2 y2 z2] format (normalized [0, 1])

    Kwargs:
        size (Tuple) the size of the image

    Returns:
        image (Array[*`size`])
    """
    img = np.zeros(size)

    for fill, box in enumerate(boxes, 1):

        x1, y1, z1, x2, y2, z2 = box
        img[x1:x2, y1:y2, z1:z2] = fill

    return img
