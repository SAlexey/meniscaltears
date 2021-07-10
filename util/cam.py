import os
import imageio
import torch
import tempfile
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce
from functools import partial


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


class GradCAM(nn.Module):
    def __init__(
        self,
        model,
        target_layer,
        use_cuda=False,
        reshape_transform=None,
        postprocess=None,
    ):
        super().__init__()
        self.model = model.eval()
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        self.reshape_transform = reshape_transform

        # postprocess outputs in any way (eg reshape)
        self.postprocess = postprocess

        if self.use_cuda:
            self.model = self.model.cuda()

        self.activations_and_gradients = ActivationsAndGradients(model, target_layer)

    def get_cam_weights(self, input, target_category, activations, gradients):
        return gradients.mean(dim=(2, 3, 4))

    def get_loss(self, output, target_category):
        loss = output[tuple(target_category)]
        # for i in range(len(target_category)):
        #     loss = loss + output[i, [0, 0]]
        return loss

    def get_cam_image(self, input, target_category, activations, gradients):

        weights = self.get_cam_weights(input, target_category, activations, gradients)
        weighted_activations = weights[:, :, None, None, None] * activations

        cam = weighted_activations.max(dim=1, keepdim=True).values
        return cam

    def forward(self, input_tensor, target_category, eigen_smooth=False):

        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_gradients(input_tensor)

        if self.postprocess is not None:
            output = self.postprocess(output)

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_gradients.activations[-1].cpu().data
        grads = self.activations_and_gradients.gradients[-1].cpu().data

        cam = self.get_cam_image(input_tensor, target_category, activations, grads)

        result = F.interpolate(cam, input_tensor.shape[-3:])

        return result


class MenisciCAM(GradCAM):

    """
    Class Activation Mapping (CAM) for menisci predictions
    """

    def get_cam_weights(self, input, meniscus, activations, gradients):
        return gradients.mean(dim=(2, 3, 4))

    def get_loss(self, output, meniscus):

        labels = output["labels"][:, meniscus]
        if "boxes" in output.keys():
            boxes = output["boxes"][:, meniscus]
            loss_boxes = boxes.sum(dim=1)

        loss_labels = labels.sum(dim=1)

        loss = loss_labels  # + loss_boxes # <- depending on boxes ?

        return loss

    def get_cam_image(self, input, target_category, activations, gradients):

        weights = self.get_cam_weights(input, target_category, activations, gradients)
        weighted_activations = weights[:, :, None, None, None] * activations

        cam = weighted_activations.max(dim=1, keepdim=True).values
        return cam


def to_gif(img, heatmap, out_path):
    tmp_files = []
    desc = []
    
    for n in range(img.shape[2]):
        fd, path = tempfile.mkstemp(suffix=".png")
        tmp_files.append(path)
        desc.append(fd)
        fig = plt.figure()

        plt.imshow(img.detach().cpu().numpy()[0,0,n], 'gray', interpolation='none')
        plt.imshow(heatmap[n], 'jet', interpolation='none', alpha=0.25)
        plt.savefig(path)
        plt.close()

    images = []
    for path, fd  in zip(tmp_files,desc):
        images.append(imageio.imread(path))

        os.remove(path)
        os.close(fd)
    
    imageio.mimsave(out_path, images)
