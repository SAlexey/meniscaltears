from collections import deque
import json
from numbers import Number
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
from einops.einops import rearrange



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


def class_getter(obj):
    return obj.__class__


def _is_numeric(obj):
    if isinstance(obj, Number):
        return True
    elif _is_sequence(obj) and not isinstance(obj, (str, dict)):
        return all([_is_numeric(each) for each in obj])
    else:
        return False


def _reduce(s: Sequence[Dict[str, torch.Tensor]]):
    """
    Convert a sequence of dictionaries containing tensors
    to a dictionary of tensors where the tensors are
    a result of concatenation


    Example:

        s = [
            {"k0": tensor([[1,2,3]]), "k1": tensor([[2, 5]])},
            {"k0": tensor([[1,2,3]]), "k1": tensor([[2, 5]])},
            {"k0": tensor([[1,2,3]]), "k1": tensor([[2, 5]])},
        ]

        output = _reduce(s)

        output:
            {
                "k0": tensor([
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ]),
                "k1": tensor([
                     [2, 5],
                     [2, 5],
                     [2, 5],
                ])
            }

    """

    result = defaultdict(list)

    for collection in s:

        for key, value in collection.items():
            if value.ndim == 0:
                result[key].append(value)
            else:
                # assuming every value is of shape (BS, *) [BS = batch size]
                # where BS is variable and * is arbitrary number of dims
                result[key].extend([item for item in value.unbind(0)])
    for key, value in result.items():
        pattern = "list " + " ".join((f"d{dim}" for dim in range(value[0].ndim)))
        try:
            result[key] = rearrange(value, f"{pattern} -> {pattern}")
        except:
            print(key, value)

    return result


def _get_model_device(model):
    return next(model.parameters()).device


def _to_device(
    input: torch.Tensor,
    target: Dict[str, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
):
    return input.to(device), {k: v.to(device) for k, v in target.items()}


class LoggableMixin(object):
    def get_log(self):
        raise NotImplementedError


class SmoothedValue(object):
    def __init__(self, initial=None, window=20):
        initial = [] if initial is None else [initial]
        self.values = deque(initial, maxlen=window)

    @property
    def value(self):
        if self.values:
            return self.values[-1]
        return np.nan

    def max(self, if_empty=np.nan):
        if len(self.values):
            return max(self.values)
        return if_empty

    def mean(self, if_empty=np.nan):
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


class EarlyStopping(object):

    def __init__(self, *, tolerance=1e-5, patience=7, warmup=10, minimize=True, name="criterion"):
        self.name = name 
        self.minimize = minimize 
        self.tolerance = tolerance
        self.warmup = warmup # number of epochs to warmup
        self.patience = patience
        self._best = np.inf
        self._last = 0

        if not self.minimize:
            self._best *= -1

    def checkpoint(self, model, epoch, optimizer=None, scheduler=None, early_stop=False, name="checkpoint.ckpt", **kwargs):

        state_dict = {
            "epoch": epoch,
            "early_stop": early_stop,
            "model": model.state_dict(),
            "best_epoch": self._last,
            f"best_{self.name}": self._best,
            **kwargs
        }

        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()

        if scheduler is not None: 
            state_dict["scheduler"] = scheduler.state_dict()

        self.save(state_dict, name)

    def _save_best(self, model):
        best, epoch = self.best
        info = {
            "epoch": epoch,
            f"best_{self.name}": best
        }
        self.save(model.state_dict(), "best_model.pt", **info)        
        

    @staticmethod
    def save(obj, name, **info):

        path = Path(name)
        torch.save(obj, name)
        
        if info:
            misc = path.with_suffix(".json")
            with misc.open("w") as fh:
                json.dump(info, fh)

    @property
    def best(self):
        return self._best, self._last

    
    def stop(self, value, epoch, model):
        stop = False
        best, last = self.best
        value_diff = value - best
        epoch_diff = epoch - max(last, self.patience)

        # if the change is insignificant
        if abs(value_diff) < self.tolerance:
            stop = True

        # if the patience is out
        if epoch_diff > self.patience:
            stop = True

        # update best values
        if self.minimize:
            if value_diff < 0:
                self._best = value 
                self._last = epoch 
                self._save_best(model) 
                
        else:
            if value_diff > 0:
                self._best = value
                self._last = epoch
                self._save_best(model)
               
        return stop


    def __call__(self, value, model, epoch):   
        return self.stop(value, epoch, model)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False

    elif tensor_list[0].ndim == 4:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, d, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, d, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[
                : img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]
            ].copy_(img)
            m[: img.shape[1], : img.shape[2], : img.shape[3]] = False

    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


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
