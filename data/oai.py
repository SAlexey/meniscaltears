#%%
import os
from util.box_ops import box_cxcywh_to_xyxy, denormalize_boxes
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

import SimpleITK as sitk
from scipy import ndimage
import math

from util.box_ops import box_cxcywh_to_xyxy, denormalize_boxes
from torch.utils.data import DataLoader
from .transforms import *


class CropDataset(Dataset):
    """
    Base class for all the datasets
    Args:
        root (str|Path-like): path to inputs
        anns (str|Path-like): path to a json annotation file
    Kwargs:
        transforms (optional, Sequence[Callable]): transforms applied to the inputs
        size (optional, Tuple): desired img size


    Notes:
        annotations: must be a json file that loads into a dictionary with unique image ids as keys
        example:
        {
            image_id_0: ann_0,
            image_id_1: ann_1
            ...
        }
    """

    def __init__(self, root, anns, transforms=None, size=None, tse=False):
        self.root = root
        self.train = "train" in str(anns)
        self.tse = tse
        with open(anns) as fh:
            anns = json.load(fh)

        # filter annotations where there are two boxes
        self.anns = dict()
        for ann in [
            ann for ann in anns if len(ann["boxes"]) == 2 and all(ann["boxes"])
        ]:
            self.anns[ann["image_id"]] = ann

        self.keys = [
            ann["image_id"]
            for ann in anns
            if len(ann["boxes"]) == 2 and all(ann["boxes"])
        ]

        if not size:
            self._img_size()
        else:
            assert size, "Please specify voxel size"
            self.img_size = size

        self._targets()
        self.transform = transforms

    def _img_size(self):
        dividable_eight = lambda x: math.ceil(x / 16) * 16
        size_factor = 1.05
        mins = []
        maxs = []
        for ann in self.anns.values():
            mins.append(np.min(np.array(ann["boxes"]), axis=0)[:3])
            maxs.append(np.min(np.array(ann["boxes"]), axis=0)[3:])
        maxs = np.vstack(maxs)
        mins = np.vstack(mins)
        if self.tse:
            percent = 99
        else:
            percent = 100
        max_roi = size_factor * np.percentile(maxs - mins, percent, axis=0)
        self.img_size = tuple(map(dividable_eight, max_roi))

    def _targets(self):
        self.targets = dict()
        self.pos_weight = 0

        for ann in self.anns.values():
            target = {
                "image_id": np.asarray(ann["image_id"], dtype=int),
                "patient_id": np.asarray(ann.get("patient_id", 0), dtype=int),
            }
            labels = [
                [
                    ann.get("V00MMTLA"),
                    ann.get("V00MMTLB"),
                    ann.get("V00MMTLP"),
                ],
                [
                    ann.get("V00MMTMA"),
                    ann.get("V00MMTMB"),
                    ann.get("V00MMTMP"),
                ],
            ]

            labels = np.nan_to_num(np.asarray(labels, dtype=np.float32))
            labels = (labels > 1).astype(float)

            target["labels"] = labels
            target["boxes"] = np.array(ann.get("boxes"))
            self.pos_weight += labels
            self.targets[ann["image_id"]] = target

        self.pos_weight = torch.as_tensor(
            (len(self) - self.pos_weight) / self.pos_weight
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        input = self._get_input(key)
        target = self._get_target(key)

        if self.transform is not None:
            input, target = self.transform(input, target, size=self.img_size)

        ann = self.anns[key]

        # flip left to right.
        if ann["side"] == "left":
            # assume by now input is a torch.Tensor[ch d h w]
            input = input.flip(1)

        return input, target

    def _get_input(self, key):
        read = self._get_reader()
        path = os.path.join(self.root, f"{key}.npy")
        return np.expand_dims(read(path), 0).clip(0, 255).astype(float)

    def _get_reader(self):
        return np.load

    def _get_target(self, key):
        return self.targets[key]


class DatasetBase(Dataset):

    """
    Base class for all the datasets
    Args:
        root (str|Path-like): path to inputs
        anns (str|Path-like): path to a json annotation file

    Kwargs:
        transforms (optional, Sequence[Callable]): transforms applied to the inputs


    Notes:
        annotations: must be a json file that loads into a dictionary with unique image ids as keys
        example:
        {
            image_id_0: ann_0,
            image_id_1: ann_1
            ...
        }
    """

    def __init__(
        self,
        root,
        anns,
        transforms=None,
    ):
        self.root = root
        with open(anns) as fh:
            anns = json.load(fh)

        # filter annotations where there are two boxes
        self.anns = [
            ann for ann in anns if len(ann["boxes"]) == 2 and all(ann["boxes"])
        ]

        self.transform = transforms

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        input = self._get_input(idx)
        target = self._get_target(idx)

        if self.transform is not None:
            input, target = self.transform(input, target)

        return input, target

    def _get_input(self, idx):
        ann = self.anns[idx]
        read = self._get_reader(ann)
        path = os.path.join(self.root, f"{ann['image_id']}.npy")
        return read(path)

    def _get_reader(self, idx):
        return np.load

    def _get_target(self, idx):
        return self.anns[idx]


class MOAKSDataset(DatasetBase):
    def __init__(
        self,
        root,
        anns,
        *args,
        labelling: str = "region-tear",
        transforms=None,
        **kwargs,
    ):

        self.root = root
        with open(anns) as fh:
            anns = json.load(fh)

        # filter annotations where there are two boxes
        self.transform = transforms
        self.anns = []
        self.targets = []
        self.labelling = labelling
        # FIXME: position labels for different annotation types!
        self.pos_weight = None if labelling.endswith("moaks") else 0

        assert labelling in {
            "region-tear",
            "region-anomaly",
            "region-moaks",
            "meniscus-tear",
            "meniscus-anomaly",
            "global-tear",
            "global-anomaly",
        }

        for ann in anns:

            if not all(ann["boxes"]) or len(ann["boxes"]) != 2:
                continue

            target = {
                "image_id": torch.as_tensor(ann["image_id"], dtype=int),
                "patient_id": torch.as_tensor(ann.get("patient_id", 0), dtype=int),
            }

            labels = [
                [
                    ann.get("V00MMTLA", 0.0),
                    ann.get("V00MMTLB", 0.0),
                    ann.get("V00MMTLP", 0.0),
                ],
                [
                    ann.get("V00MMTMA", 0.0),
                    ann.get("V00MMTMB", 0.0),
                    ann.get("V00MMTMP", 0.0),
                ],
            ]

            labels = torch.from_numpy(np.nan_to_num(np.asarray(labels).astype(float)))

            resolution, detection = labelling.split("-")

            if resolution == "meniscus":
                labels = labels.max(-1, keepdim=True).values
            elif resolution == "global":
                labels = labels.max().unsqueeze(0)

            if detection == "tear":
                labels = (labels > 1).float()
            elif detection == "anomaly":
                labels = (labels >= 1).float()

            if self.pos_weight is not None:
                self.pos_weight += labels

            target["labels"] = labels
            target["boxes"] = torch.as_tensor(ann.get("boxes"))

            self.targets.append(target)
            self.anns.append(ann)

        if self.pos_weight is not None:
            self.pos_weight = torch.as_tensor(
                (len(self) - self.pos_weight) / self.pos_weight
            )

    def __len__(self):
        return len(self.anns)

    def _get_input(self, idx):
        ann = self.anns[idx]
        input = np.load(os.path.join(self.root, f"{ann['image_id']}.npy"))
        input = torch.from_numpy(input).clip(0, 255).unsqueeze(0)
        return input.float()

    def _get_target(self, key):
        return self.targets[key]

    def __getitem__(self, idx):
        input = self._get_input(idx)
        target = self._get_target(idx)
        ann = self.anns[idx]

        # flip left to right.
        # see README.md
        if ann["side"] == "left":
            # assume by now input is a torch.Tensor[ch d h w]
            # _, d, *_ = input.size()

            input = input.flip(1)
            target["boxes"] = target["boxes"].flip(0)

        if self.transform is not None:
            input, target = self.transform(input, target)

        return input, target


class DICOMDataset(DatasetBase):
    reader = sitk.ImageSeriesReader()

    def _get_reader(self, key):
        tgt = self._get_target(key)
        file_names = self.reader.GetGDCMSeriesFileNames(tgt["dicom_dir"])
        self.reader.SetFileNames(file_names)
        return self.reader

    def _get_input(self, key):
        img = self._get_reader(key).Execute()
        img = sitk.GetArrayFromImage(img)
        return img


class DICOMDatasetMasks(DICOMDataset):
    def _get_reader(self, key):
        tgt = self._get_target(key)
        path = Path(tgt["dicom_dir"].replace("/vis/scratchN/oaiDataBase/", ""))
        path = self.root / path / "Segmentation"
        file_names = self.reader.GetGDCMSeriesFileNames(path)
        self.reader.SetFileNames(file_names)
        return self.reader

    def __getitem__(self, idx):
        key = self.keys[idx]
        mask = self._get_input(key)
        boxes = []

        objects = ndimage.find_objects(mask)
        for label, obj in enumerate(objects, 1):
            if obj is not None and label in (5, 6):
                xs, ys, zs = obj
                box = [
                    (xs.start + xs.stop) / 2.0,
                    (ys.start + ys.stop) / 2.0,
                    (zs.start + zs.stop) / 2.0,
                    xs.stop - xs.start,
                    ys.stop - ys.start,
                    zs.stop - zs.start,
                ]
                boxes.append(box)

        mask = torch.from_numpy(mask)
        boxes = torch.as_tensor(boxes) / torch.as_tensor(mask.shape).repeat(1, 2)
        return mask.unsqueeze(0), boxes


class MixDataset(Dataset):
    def __init__(self, root, anns, anns_tse, train=False, p=0.1, transforms=None):
        self.train = train
        self.p = p
        self.dess = MOAKSDataset(
            root,
            anns,
            binary=True,
            multilabel=True,
            transforms=Compose(
                (
                    ToTensor(),
                    RandomInvert(0.15),
                    Resize((160, 384, 384)),
                    Normalize(mean=(0.4945), std=(0.3782,)),
                )
            ),
        )
        self.tse = MOAKSDataset(
            root,
            anns_tse,
            binary=True,
            multilabel=True,
            transforms=Compose(
                (
                    ToTensor(),
                    RandomInvert(0.15),
                    Resize((160, 384, 384)),
                    Normalize(mean=(0.359,), std=(0.278,)),
                )
            ),
        )

        self.transform = transforms
        if self.train:
            self.pos_weight = (self.dess.pos_weight + self.tse.pos_weight) / 2

    def __len__(self):
        return len(self.dess) + len(self.tse) - 1

    def __getitem__(self, idx):

        dess = idx < len(self.dess)

        if dess:
            img, tgt = self.dess[idx]
        else:
            img, tgt = self.tse[idx - len(self.dess)]

        if self.train and random() <= self.p:

            if dess:
                other, target = self.tse[randint(0, len(self.tse) - 1)]

            else:
                other, target = self.dess[randint(0, len(self.dess) - 1)]

            alpha = random()
            beta = 1 - alpha
            sign = 1 if random() <= 0.5 else -1

            img = (alpha * img) + (sign * beta * other)

            tgt["labels"] = (alpha * tgt["labels"]) + (sign * beta * target["labels"])

            # renormalize

            img = img - img.mean()
            img = img / img.std()

            if beta > 0.5:
                tgt["boxes"] = target["boxes"]

        if self.transform is not None:
            boxes = box_cxcywh_to_xyxy(tgt["boxes"])
            tgt["boxes"] = denormalize_boxes(boxes, (160, 384, 384))
            img, tgt = self.transform(img, tgt)

        return img, tgt


def build(
    data_dir,
    anns_dir,
    labelling,
    limit_train_items=False,
    limit_val_items=False,
    limit_test_items=False,
    train=False,
    isotrope=False,
):

    root = Path("/scratch/htc/ashestak")

    if not root.exists():
        root = Path("/scratch/visual/ashestak")

    if not root.exists():
        raise ValueError(f"Invalid root directory: {root}")

    data_dir = root / data_dir
    anns_dir = root / anns_dir

    assert data_dir.exists(), "Provided data directory doesn't exist!"
    assert anns_dir.exists(), "Provided annotations directory doesn't exist!"

    to_tensor = ToTensor()
    center_crop = CenterCropVolume((160, 320, 320))

    tse = "tse" in str(anns_dir)

    if isotrope:
        output_size = (320, 320, 320)
    else:
        output_size = (44, 320, 320) if tse else (160, 320, 320)

    resize = Resize(output_size)
    b_size, w_size = 11, 7

    if tse:
        w_size, b_size = b_size, w_size
        normalize = Normalize(mean=(0.21637,), std=(0.18688,))
    else:
        normalize = Normalize(mean=(0.4945,), std=(0.3782,))

    tophat = TopHatFilter(b_size, w_size)

    transforms = Compose([to_tensor, center_crop, resize, normalize, tophat])

    if train:

        train_transforms = Compose(
            [
                to_tensor,
                RandomResizedBBoxSafeCrop(p=0.5, bbox_safe=True),
                AugSmoothTransform(p=0.5),
                center_crop,
                resize,
                normalize,
                tophat,
            ]
        )

        dataset_train = MOAKSDataset(
            data_dir,
            anns_dir / "train.json",
            labelling=labelling,
            transforms=train_transforms,
        )

        if limit_train_items:
            dataset_train.anns = dataset_train.anns[:limit_train_items]

        dataset_val = MOAKSDataset(
            data_dir,
            anns_dir / "val.json",
            labelling=labelling,
            transforms=transforms,
        )

        if limit_val_items:
            dataset_val.anns = dataset_val.anns[:limit_val_items]

        return dataset_train, dataset_val

    else:

        dataset_test = MOAKSDataset(
            data_dir,
            anns_dir / "test.json",
            labelling=labelling,
            transforms=transforms,
        )

        if limit_test_items:
            dataset_test.anns = dataset_test.anns[:limit_test_items]

        return dataset_test


# %%
