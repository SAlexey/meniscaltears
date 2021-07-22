from data.transforms import AddGaussianNoise
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
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from random import random, randrange
from data.transforms import crop_volume


class AugSmoothTransform(T.Compose):
    def __init__(self, p=0.5):
        self.p = p
        self.noise = torch.distributions.Normal(0.1, 0.5)
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


class HeatmapGifOverlayMixin(object):

    NAMES = [("L", "M")]

    def to_gif(self, img, heatmap, name, cam_type="grad"):
        tmp_files = []
        desc = []
        cmap = "hot"
        alpha = np.ones(len(img))

        if heatmap.any():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = (heatmap - heatmap.mean()) / heatmap.std()

            if cam_type == "grad":
                p = np.percentile(heatmap, 98)
                alpha = np.ones(heatmap.shape)
                alpha[heatmap < p] = 0
                alpha[heatmap >= p] = 0.5
                heatmap[heatmap < p] = p
                cmap = "jet"
            else:
                p = np.percentile(heatmap, q=99)
                alpha = np.ones(heatmap.shape)
                alpha[heatmap < p] = 0
                alpha[heatmap >= p] = 0.7
                heatmap[heatmap < p] = p
        else:
            alpha *= 0

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


class ActivationsAndGradients:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class GradCAM(nn.Module, HeatmapGifOverlayMixin):
    def __init__(self, model, target_layer, use_cuda=False, postprocess=None):
        super().__init__()
        self.model = model.eval()
        self.layer = target_layer
        self.use_cuda = use_cuda
        self.postprocess = postprocess
        if self.use_cuda:
            model.cuda()

        self.activations_and_gradients = ActivationsAndGradients(model, target_layer)

    def get_cam_weights(self, input, key, index, activations, gradients):
        return gradients.mean(dim=(2, 3, 4))

    def get_loss(self, output, key, index):
        return output[key][index].sum()

    def get_cam_image(self, input, key, index, activations, gradients):

        weights = self.get_cam_weights(input, key, index, activations, gradients)
        weighted_activations = weights[:, :, None, None, None] * activations

        cam = weighted_activations.max(dim=1, keepdim=True).values
        return cam

    def forward(
        self,
        input,
        output_index,
        output_key="labels",
        aug_smooth=False,
        num_passes=15,
        save_as=False,
        target=None,
    ):

        if self.use_cuda:
            input = input.cuda()

        output = self.activations_and_gradients(input)

        if self.postprocess is not None:
            output = self.postprocess(output)

        self.model.zero_grad()
        loss = self.get_loss(output, output_key, output_index)
        loss.backward(retain_graph=True)

        activations = self.activations_and_gradients.activations[-1].cpu().data
        grads = self.activations_and_gradients.gradients[-1].cpu().data

        cam = self.get_cam_image(input, output_key, output_index, activations, grads)

        if aug_smooth:
            # performs a few more iterations with augmentation of the input image
            # averages cam results in the end
            # should reduce noise
            image_aug = AugSmoothTransform()
            for _ in range(num_passes):  # 5 passes
                aug_input = image_aug(input)

                output = self.activations_and_gradients(aug_input)

                if self.postprocess is not None:
                    output = self.postprocess(output)
                self.model.zero_grad()

                loss = self.get_loss(output, output_key, output_index)
                loss.backward(retain_graph=True)

                activations = self.activations_and_gradients.activations[-1].cpu().data
                grads = self.activations_and_gradients.gradients[-1].cpu().data

                cam += self.get_cam_image(
                    aug_input, output_key, output_index, activations, grads
                )

            # take the average of cams
            cam = cam / num_passes

        result = F.interpolate(
            cam, input.shape[-3:], mode="trilinear", align_corners=False
        )
        result = result.squeeze().detach().cpu().numpy()

        if save_as:
            name = save_as + ("_aug_cam" if aug_smooth else "_cam")

            self.to_gif(input.squeeze().cpu().numpy(), result, name)

        return result


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
        self.img_aug = AugSmoothTransform()

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        # update self.device
        self.device = _get_model_device(self.model)
        return super().to(*args, **kwargs)

    def get_saliency(
        self,
        input: torch.Tensor,
        key,
        index,
        regions,
        saliency=torch.abs,
    ):

        assert saliency is not None and callable(saliency)
        input.requires_grad_().retain_grad()
        self.model.zero_grad()

        output = self.model(input)

        if self.postprocess is not None:
            output = self.postprocess(output)
        out = (output[key][index] * regions).sum()
        out.backward()

        out = saliency(input.grad.data)

        return out, output

    def get_smooth_grad(
        self,
        input: torch.Tensor,
        key,
        index,
        regions,
        num_passes=50,
    ):
        input = input.to(self.device)

        grad, output = self.get_saliency(input, key, index, regions)

        grads = [grad]

        progress = range(num_passes)
        if self.progress:
            progress = tqdm(progress)

        for _ in progress:

            grads.append(self.get_saliency(self.img_aug(input), key, index, regions)[0])

        return (
            grad.detach().cpu(),
            torch.stack(grads).mean(dim=0).detach().cpu(),
            output,
        )

    def forward(
        self,
        input,
        index,
        regions,
        key="labels",
        num_passes=15,
        save_as=False,
        boxes=None,
    ):
        regions = regions.to(self.device)
        self.model.eval()

        assert (
            input.ndim == 5
        ), f"Expecting 5-dimensional input [bs ch d h w], got ndim={input.ndim}"

        assert input.size(0) == 1, f"Expecting batch size 1, got bs={input.size(0)}"

        vanilla_grad, smooth_grad, output = self.get_smooth_grad(
            input, key, index, regions, num_passes=num_passes
        )

        if save_as:

            inputs = []
            names = []

            if boxes is not None:

                input = input.squeeze(0).detach()
                vanilla_grad = vanilla_grad.squeeze(0)
                smooth_grad = smooth_grad.squeeze(0)

                box = box_cxcywh_to_xyxy(boxes)
                box = denormalize_boxes(box)

                box = box[index[1], [0, 3]].squeeze()

                # crop (tgt_box_zmin, 0, 0, box_depth, img_height, img_width)
                crop = box[0], 0, 0, box[1] - box[0], *input.size()[-2:]
                print("crop target box", crop)

                image, _ = crop_volume(input, crop)
                image_grad, _ = crop_volume(smooth_grad, crop)

                images = (image, image_grad)
                image_names = ("image", "smooth_grad")

                if self.vanilla:

                    images += (crop_volume(vanilla_grad, crop)[0],)
                    image_names += ("vanilla_grad",)

                inputs.append(images)
                names.append(image_names)

                # crop by output
                box = box_cxcywh_to_xyxy(output["boxes"])
                box = denormalize_boxes(box)
                box = box[index[:2]].squeeze()

                # crop only by output box
                crop = *box[:3], box[3] - box[0], box[4] - box[1], box[5] - box[2]

                print("crop output box", crop)

                image, _ = crop_volume(input, crop)
                image_grad, _ = crop_volume(smooth_grad, crop)

                images = (image, image_grad)
                image_names = ("image_crop", "smooth_grad_crop")

                if self.vanilla:

                    images += (crop_volume(vanilla_grad, crop)[0],)
                    image_names += ("vanilla_grad_crop",)

                inputs.append(images)
                names.append(image_names)

            for input, name in zip(inputs, names):

                image = input[0].squeeze().cpu().numpy()
                image_name = name[0]

                grad_images = input[1:]
                grad_names = name[1:]

                # save image only
                zero_grad = np.zeros_like(image)
                self.to_gif(image, zero_grad, f"{save_as}_{image_name}", cam_type="gradient")

                for grad, grad_name in zip(grad_images, grad_names):
                    grad = grad.squeeze().cpu().numpy()

                    # save image with overlay gradient

                    self.to_gif(
                        image, grad, f"{save_as}_{grad_name}_overlay", cam_type="gradient"
                    )

        return smooth_grad
