import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, denormalize_boxes
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.std import tqdm
from .misc import _get_model_device
from torch.distributions import Normal, Uniform
import tempfile
import imageio
import matplotlib.pyplot as plt
import os


class HeatmapGifOverlayMixin(object):

    NAMES = [("L", "M"), ("AH", "B", "PH")]

    def to_gif(self, img, heatmap, name):
        tmp_files = []
        desc = []
        cmap = "hot"
        alpha = [0.3] * img.shape[0]

        heatmap = (heatmap - heatmap.mean()) / heatmap.std()
        heatmap = heatmap / heatmap.max()
        p = np.percentile(heatmap, q=99.5)

        heatmap = np.where(heatmap > p, heatmap, 0.0)

        for n in range(img.shape[0]):
            fd, path = tempfile.mkstemp(suffix=".png")
            tmp_files.append(path)
            desc.append(fd)
            _ = plt.figure()

            plt.imshow(img[n], "gray", interpolation="none")
            plt.imshow(heatmap[n], cmap, interpolation="none", alpha=alpha[n])
            plt.savefig(path)
            plt.close()

        images = []
        for path, fd in zip(tmp_files, desc):
            images.append(imageio.imread(path))

            os.remove(path)
            os.close(fd)

        if not name.endswith(".gif"):
            name = f"{name}.gif"

        imageio.mimsave(name, images)


class GradCAM(nn.Module):
    def __init__(self, model, target_layer, use_cuda=False, postprocess=None):
        super().__init__()
        self.model = model.eval()
        self.layer = target_layer
        self.use_cuda = use_cuda
        self.postprocess = postprocess

        if self.use_cuda:
            model.cuda()

    def get_cam_weights(self, input, key, index, activations, gradients):
        return gradients.mean(dim=(2, 3, 4))

    def get_loss(self, output, key, index):
        return output[key][index]

    def get_cam_image(self, input, key, index, activations, gradients):

        weights = self.get_cam_weights(input, activations, gradients)
        weighted_activations = weights[:, :, None, None, None] * activations

        cam = weighted_activations.max(dim=1, keepdim=True).values
        return cam

    def forward(self, input, key, index, aug_smooth=False):

        if self.use_cuda:
            input = input.cuda()

        output = self.activations_and_gradients(input)

        if self.postprocess is not None:
            output = self.postprocess(output)

        self.model.zero_grad()
        loss = self.get_loss(output, key, index)
        loss.backward(retain_graph=True)

        activations = self.activations_and_gradients.activations[-1].cpu().data
        grads = self.activations_and_gradients.gradients[-1].cpu().data

        cam = self.get_cam_image(input, key, index, activations, grads)

        result = F.interpolate(cam, input.shape[-3:], mode="trilinear")

        return result.detach().cpu().numpy()


class SmoothGradientSaliency(nn.Module, HeatmapGifOverlayMixin):
    def __init__(
        self,
        model: nn.Module,
        postprocess=None,
        # smooth grad sigma range for gaussian noize
        noise_scale=(0.1, 0.205),
        progress=True,
        vanilla=False,
        boxes=False,
    ):
        super().__init__()
        self.model = model
        self.postprocess = postprocess
        self.sg_scale = Uniform(*noise_scale)
        self.device = _get_model_device(model)
        self.progress = progress
        self.boxes = boxes
        self.vanilla = vanilla

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

        out = output["labels"][0, target_obj, target_cls]

        out += boxes.sum()
        out.backward()

        out = saliency(input.grad.data)

        return out, output

    def get_smooth_grad(
        self,
        input: torch.Tensor,
        target_obj,
        target_cls,
        num_passes=50,
    ):

        imin, imax = input.min(), input.max()
        input = input.to(self.device)

        grad, output = self.get_saliency(input, target_obj, target_cls)

        grads = [grad]
        loc = torch.tensor([0.0]).to(self.device)

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
                    input + noise,
                    target_obj,
                    target_cls,
                )[0]
            )

        return grad.detach().cpu(), torch.stack(grads).mean(dim=0).detach().cpu()

    def forward(
        self,
        input,
        target_obj,
        target_cls,
        num_passes=50,
        gif=True,
        name_prefix="",
    ):

        self.model.eval()

        assert (
            input.ndim == 5
        ), f"Expecting 5-dimensional input [bs ch d h w], got ndim={input.ndim}"

        assert input.size(0) == 1, f"Expecting batch size 1, got bs={input.size(0)}"

        vanilla_grad, smooth_grad = self.get_smooth_grad(
            input, target_obj, target_cls, num_passes=num_passes
        )

        if gif:

            input = input.squeeze().numpy()
            for prefix, grad in zip(
                ("vanilla_grad", "smooth_grad"), (vanilla_grad, smooth_grad)
            ):

                if prefix == "vanilla_grad" and not self.vanilla:
                    continue

                if name_prefix:
                    prefix = f"{prefix}_{name_prefix}"

                name = f"{prefix}_{self.NAMES[0][target_obj]}{self.NAMES[1][target_cls]}_{num_passes}p"
                grad = grad.squeeze().numpy()
                self.to_gif(input, grad, name)

        return smooth_grad
