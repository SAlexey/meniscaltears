import torch
from torch import nn
import torch.nn.functional as F
from tqdm.std import tqdm
from .misc import _get_model_device
from torch.distributions import Normal, Uniform


class SmoothGradientSaliency(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        postprocess=None,
        # smooth grad sigma range for gaussian noize
        noise_scale=(0.1, 0.205),
        progress=True,
        boxes=False,
    ):
        super().__init__()

        self.model = model
        self.postprocess = postprocess
        self.sg_scale = Uniform(*noise_scale)
        self.device = _get_model_device(model)
        self.progress = progress
        self.boxes = boxes

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        # update self.device
        self.device = _get_model_device(self.model)
        return super().to(*args, **kwargs)

    def get_saliency(
        self,
        input: torch.Tensor,
        target_obj,
        target_cls,
        saliency=torch.abs,
    ):

        assert saliency is not None and callable(saliency)
        input.requires_grad_().retain_grad()
        self.model.zero_grad()

        output = self.model(input)

        if self.postprocess is not None:
            output = self.postprocess(output)

        if self.boxes:
            assert "boxes" in output
            boxes = output["boxes"].squeeze()[target_obj]
        else:
            boxes = torch.tensor([0.0]).to(self.device)

        labels = output["labels"].squeeze()[target_obj].sigmoid()
        labels = labels.where(labels == target_cls, torch.tensor([0.0]).to(self.device))

        out = labels.sum() + boxes.sum()

        out = saliency(input.grad.data)

        return out, output

    def get_smooth_grad(
        self,
        input: torch.Tensor,
        target_obj,
        target_cls,
        num_passes=50,
        noise_mask=None,
    ):

        if noise_mask is None:
            noise_mask = torch.ones_like(input)

        assert (
            noise_mask.size() == input.size()
        ), f"Expected noise mask to be the same size as input, got input={noise_mask.size()} and mask={noise_mask.size()}"

        imin, imax = input.min(), input.max()
        input = input.to(self.device)

        grad, output = self.get_saliency(input, target_obj, target_cls)

        grads = [grad]
        loc = torch.tensor([0.0]).to(self.device)
        noise_mask = noise_mask.to(self.device)

        progress = range(num_passes)
        if self.progress:
            progress = tqdm(progress)

        for _ in progress:
            scale = self.sg_scale.sample() * (imax - imin)
            scale = scale.to(self.device)

            noise = Normal(loc, scale).sample(input.size())
            noise = noise.to(self.device).view(*input.size())

            grads.append(
                self.get_saliency(
                    input + noise * noise_mask,
                    target_obj,
                    target_cls,
                )[0]
            )

        return grad.detach().cpu(), torch.stack(grads).mean(dim=0).detach().cpu()

    def forward(self, input, target_obj, target_cls, num_passes=50):

        self.model.eval()

        assert (
            input.ndim == 5
        ), f"Expecting 5-dimensional input [bs ch d h w], got ndim={input.ndim}"

        assert input.size(0) == 1, f"Expecting batch size 1, got bs={input.size(0)}"

        vanilla_grad, smooth_grad = self.get_smooth_grad(
            input, target_obj, target_cls, num_passes=num_passes
        )

        return smooth_grad
