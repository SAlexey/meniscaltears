from collections import deque
from numbers import Number
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path


def save_as_nifty(arr, name):
    img = nib.Nifti1Image(arr.astype("uint8"), np.eye(4))
    img.to_filename(name)


def bbox_image(boxes: torch.Tensor, size=(160, 384, 384)):

    img = torch.zeros(*size)

    # boxes = denormalize_boxes(boxes, size)

    for label, box in enumerate(boxes, 1):
        img[box[0] : box[3], box[1] : box[4], box[2] : box[5]] = label

    return img.numpy()


def _is_sequence(obj):
    return hasattr(obj, "__len__") and hasattr(obj, "__iter__")


def _is_numeric(obj):
    if isinstance(obj, Number):
        return True
    elif _is_sequence(obj) and not isinstance(obj, (str, dict)):
        return all([_is_numeric(each) for each in obj])
    else:
        return False


class SmoothedValue(object):
    def __init__(self, initial=None, window=20):
        initial = [] if initial is None else [initial]
        self.values = deque(initial, maxlen=window)

    @property
    def value(self):
        return self.values[-1]

    def max(self, if_empty=None):
        if len(self.values):
            return max(self.values)
        return if_empty

    def mean(self, if_empty=None):
        if len(self.values):
            return np.mean(self.values)
        return if_empty

    def add(self, value, reduction=None):
        if _is_numeric(value):
            if _is_sequence(value):
                value = torch.as_tensor(value)
                if reduction is None:
                    raise ValueError(
                        "Provided value is a sequence and with no reduction output is ambiguous"
                    )
                value = getattr(value, reduction)().detach().item()
            self.values.append(value)
        else:
            raise ValueError("Provided value must be of numeric type!")

    def __iadd__(self, value):
        self.add(value)
        return self


class DCMSeriesReader(object):
    def __init__(self, root="/"):
        self.root = Path(root)
        self._reader = sitk.ImageSeriesReader()

    def read(self, path):

        path = self.root / path

        assert path.exists(), f"Path ({path}) does not exist!"
        assert path.is_dir(), f"Path ({path}) must be a directory!"

        file_names = self._reader.GetGDCMSeriesFileNames(str(path))
        self._reader.SetFileNames(file_names)
        return self._reader.Execute()

    @staticmethod
    def as_array(image):
        return sitk.GetArrayFromImage(image)

    @staticmethod
    def as_tensor(image):
        return torch.as_tensor(DCMSeriesReader.as_array(image))

    def get_image(self, path):
        return self.read(self.root / path)

    def get_array(self, path):
        image = self.read(path)
        return self.as_array(image)

    def get_tensor(self, path):
        image = self.read(path)
        return self.as_tensor(image)

    def __call__(self, path):
        return self.read(self.root / path)


class DCMSeries(object):
    def __init__(self, root="/", path=None):

        self.path = path
        self.image = None

        self.reader = DCMSeriesReader(root)

        if path is not None:
            self.image = self.reader.read(path)

    @property
    def array(self):
        return self.reader.as_array(self.image)

    @property
    def tensor(self):
        return self.reader.as_tensor(self.image)
