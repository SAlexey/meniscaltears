import os
import imageio
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce
from functools import partial
from torch.nn import ReLU


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

    def forward(self, input_tensor, target_category, region=None, eigen_smooth=False):

        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_gradients(input_tensor)

        if self.postprocess is not None:
            output = self.postprocess(output)

        self.model.zero_grad()
        loss = self.get_loss(output, target_category, region)
        loss.backward(retain_graph=True)

        activations = self.activations_and_gradients.activations[-1].cpu().data
        grads = self.activations_and_gradients.gradients[-1].cpu().data

        cam = self.get_cam_image(input_tensor, target_category, activations, grads)

        result = F.interpolate(cam, input_tensor.shape[-3:], mode="trilinear")

        return result


class MenisciCAM(GradCAM):

    """
    Class Activation Mapping (CAM) for menisci predictions
    """

    def get_cam_weights(self, input, meniscus, activations, gradients):
        return gradients.mean(dim=(2, 3, 4))

    def get_loss(self, output, meniscus, region=None):

        labels = output["labels"][:, meniscus]
        if "boxes" in output.keys():
            boxes = output["boxes"][:, meniscus]
            loss_boxes = boxes.sum(dim=1)

        if region:
            loss_labels = labels[:,region]
        else:
            loss_labels = labels.sum(dim=1)

        loss = loss_labels  # + loss_boxes # <- depending on boxes ?

        return loss

    def get_cam_image(self, input, target_category, activations, gradients):

        weights = self.get_cam_weights(input, target_category, activations, gradients)
        weighted_activations = weights[:, :, None, None, None] * activations

        cam = weighted_activations.max(dim=1, keepdim=True).values
        return cam


def to_gif(img, heatmap, out_path, cam_type="grad"):
    tmp_files = []
    desc = []
    assert cam_type in ["grad", "saliency", "back"]

    if cam_type=="grad":
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        alpha = np.ones(heatmap.shape)
        alpha[heatmap<0.1] = 0
        alpha[heatmap>=0.1] = .5
        heatmap[heatmap<0.1] = 0.1
        cmap = "jet"
    elif cam_type=="saliency":
        alpha = [.5] * img.shape[2]
        cmap = "hot"
    elif cam_type == "back":
        heatmap = (heatmap - heatmap.mean())/(heatmap.std())
        alpha = np.ones(heatmap.shape)
        percentile = np.percentile(heatmap, 99)
        alpha[heatmap<percentile] = 0
        alpha[heatmap>=percentile] = .5
        heatmap[heatmap<percentile] = percentile
        cmap="hot"


    for n in range(img.shape[2]):
        fd, path = tempfile.mkstemp(suffix=".png")
        tmp_files.append(path)
        desc.append(fd)
        fig = plt.figure()

        plt.imshow(img.detach().cpu().numpy()[0,0,n], 'gray', interpolation='none')
        plt.imshow(heatmap[n], cmap, interpolation='none', alpha=alpha[n])
        plt.savefig(path)
        plt.close()

    images = []
    for path, fd  in zip(tmp_files,desc):
        images.append(imageio.imread(path))

        os.remove(path)
        os.close(fd)
    
    imageio.mimsave(out_path, images)


class MenisciSaliency(nn.Module):
    """
    Saliency Mapping for menisci predictions
    """

    def __init__(
        self,
        model,
        use_cuda=False,
        postprocess=None
    ):
        super().__init__()
        self.model = model.eval()
        self.use_cuda = use_cuda
        self.postprocess = postprocess

        if self.use_cuda:
            self.model = self.model.cuda()


    def forward(self, input_tensor, target_category, label):
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        input_tensor.requires_grad_()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if self.postprocess is not None:
            output = self.postprocess(output)
        output= output["labels"].squeeze()
        output = output[target_category, label]

        output.backward()

        saliency = input_tensor.grad.data.abs()
        return saliency


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from a given image
    """
    def __init__(
        self, 
        model, 
        use_cuda=False,
        postprocess=None,
        logging=None
        ):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.use_cuda = use_cuda
        self.postprocess = postprocess

        if self.use_cuda:
            self.model = self.model.cuda()

        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.backbone._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.backbone._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def forward(self, input_image, target_label, region):
        if self.use_cuda:
            input_image = input_image.cuda()
        # Zero gradients
        input_image.requires_grad_()
        self.model.zero_grad()
        

        output = self.model(input_image)
        # Target for backprop
        if self.postprocess is not None:
            output = self.postprocess(output)
        output= output["labels"].squeeze()
        output = output[target_label, region]
        # Backward pass
        output.backward()
    
        return self.gradients
