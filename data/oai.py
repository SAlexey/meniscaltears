import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

import SimpleITK as sitk
from scipy import ndimage


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
            self.anns = json.load(fh)

        self.keys = sorted(
            [key for key in self.anns.keys() if len(self.anns[key]["boxes"]) == 2]
        )

        self.transform = transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        ins = self._get_input(key)
        tgt = self._get_target(key)

        if self.transform is not None:
            ins, tgt = self.transform(ins, tgt)

        return ins, tgt

    def _get_input(self, key):
        read = self._get_reader(key)
        path = os.path.join(self.root, f"{key}.npy")
        return read(path)

    def _get_reader(self, key):
        return np.load

    def _get_target(self, key):
        return self.anns[key]


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


class MOAKSDataset(DatasetBase):
    def _get_input(self, key):
        input = super()._get_input(key)
        return np.expand_dims(input, 0)

    def _get_target(self, key):
        ann = super()._get_target(key)
        return {
            "image_id": int(key),
            "patient_id": ann.get("patient_id", 0),
            "boxes": ann.get("boxes", []),
            "side": int(ann.get("side") == "right"),
            "labels": (ann.get("MED", 0), ann.get("LAT", 0)),
        }

    def __getitem__(self, idx):
        img, tgt = super().__getitem__(idx)

        # flip left to right
        if not tgt["side"]:
            tgt = tgt.copy()
            img = img.flip(1)
            tgt["boxes"] = tgt["boxes"].flip(0)
            tgt["labels"] = tgt["labels"].flip(0)
        return img, tgt


class MOAKSDatasetBinaryMonolabel(DatasetBase):
    def _get_input(self, key):
        input = super()._get_input(key)
        return np.expand_dims(input, 0)

    def _get_target(self, key):
        ann = super()._get_target(key)
        return {
            "image_id": int(key),
            "patient_id": ann.get("patient_id", 0),
            "boxes": ann.get("boxes", []),
            "side": int(ann.get("side") == "right"),
            "labels": (ann.get("MED", 0), ann.get("LAT", 0)),
        }

    def __getitem__(self, idx):
        img, tgt = super().__getitem__(idx)

        # flip left to right
        if not tgt["side"]:
            tgt = tgt.copy()
            img = img.flip(1)
            tgt["boxes"] = tgt["boxes"].flip(0)
            tgt["labels"] = tgt["labels"].flip(0)
        return img, tgt


class MOAKSDatasetBinaryMultilabel(MOAKSDatasetBinaryMonolabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pos_weight = []

        for key in self.keys:
            ann = self.anns[key]

            labels = np.nan_to_num(
                np.array(
                    [
                        [
                            ann["V00MMTMA"],
                            ann["V00MMTMB"],
                            ann["V00MMTMP"],
                        ],
                        [
                            ann["V00MMTLA"],
                            ann["V00MMTLB"],
                            ann["V00MMTLP"],
                        ],
                    ],
                    dtype=np.float,
                ),
            )

            if ann["side"] == "left":
                pos_weight.append(np.flip(labels, 0).flatten())
            else:
                pos_weight.append(labels.flatten())

            ann["labels"] = (labels > 1).astype(float)

        pos_weight = np.vstack(pos_weight)
        count = pos_weight.shape[0]
        pos_weight = (pos_weight > 1).sum(0)
        pos_weight = (count - pos_weight) / pos_weight
        self.pos_weight = torch.from_numpy(pos_weight.reshape(2, 3))

    def _get_target(self, key):
        tgt = super()._get_target(key)
        ann = self.anns[key]
        tgt["labels"] = ann["labels"]
        return tgt
