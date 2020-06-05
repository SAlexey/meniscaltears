import torch
from torch import FloatTensor, optim, utils, autograd, nn, cuda, stack
import SimpleITK as sitk
from torchvision import models as models_, transforms
import pandas as pd 
import numpy as np
import glob
import os 
from tqdm import tqdm, notebook
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from IPython.display import clear_output
from scipy.signal import savgol_filter
import csv
import pdb


alexnet = models_.alexnet(pretrained=True)
class Model1(nn.Module):
    """
    Using AlexNet feature layer as backbone but removed all max pooling layers
    gap and clasification inpired my MRNet from stanford
    """
    def __init__(self):
        super(Model1, self).__init__()
        self.features = nn.Sequential(
            *list(layer for layer in alexnet.features.children() 
                if not isinstance(layer, nn.MaxPool2d))        
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze()
        x = x.max(0, keepdim=True)[0]
        x = self.classify(x)
        return x
    
class Model2(nn.Module):
    
    """
    Same Model as Model1 but this time inputs are paddded with zeros to fit expected size
    while the image is placed in the top left corner
    """
    
    def __init__(self):
        super(Model2, self).__init__()
        self.pad = nn.ZeroPad2d((0, 64, 0, 176))
        self.features = nn.Sequential(*list(layer for layer in alexnet.features.children()))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pad(x)
        x = self.features(x)
        x = self.gap(x).squeeze()
        x = x.max(0, keepdim=True)[0]
        x = self.classify(x)
        return x
    
class Model3(nn.Module):
    
    """
    Same as Model2 but this time reflection padding is used 
    while the image is placed in the middle
    """
    
    def __init__(self):
        super(Model3, self).__init__()
        self.pad = nn.ReflectionPad2d((88, 88, 32, 32))
        self.features = nn.Sequential(
            *list(layer for layer in alexnet.features.children())
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pad(x)
        x = self.features(x)
        x = self.gap(x).squeeze()
        x = x.max(0, keepdim=True)[0]
        x = self.classify(x)
        return x