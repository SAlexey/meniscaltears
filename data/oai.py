#%%
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
    def __init__(self, *args, binary=True, multilabel=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = []
        self.pos_weight = 0 if binary else None

        for ann in self.anns:
            target = {
                "image_id": np.asarray(ann["image_id"], dtype=int),
                "patient_id": np.asarray(ann["patient_id"], dtype=int),
            }

            if multilabel:
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
            else:
                labels = [ann.get("LAT", 0), ann.get("MED", 0)]

            labels = np.nan_to_num(np.asarray(labels, dtype=np.float32))

            if binary:
                labels = (labels >= 1).astype(float)
                self.pos_weight += labels

            target["labels"] = labels
            target["boxes"] = np.array(ann.get("boxes"))

            self.targets.append(target)

        if self.pos_weight is not None:
            self.pos_weight = torch.as_tensor(
                (len(self) - self.pos_weight) / self.pos_weight
            )

    def _get_input(self, key):
        input = super()._get_input(key)
        return np.expand_dims(input, 0).clip(0, 255).astype(float)

    def _get_target(self, key):
        return self.targets[key]

    def __getitem__(self, idx):
        input, target = super().__getitem__(idx)
        ann = self.anns[idx]

        # flip left to right.
        # see README.md
        if ann["side"] == "left":
            # assume by now input is a torch.Tensor[ch d h w]
            input = input.flip(1)
            target["boxes"] = target["boxes"].flip(0)
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
