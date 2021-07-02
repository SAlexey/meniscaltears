import SimpleITK as sitk
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch
from torch import nn, optim, utils
from utils.train import *
from torchvision import transforms, models
from pathlib import Path
from skimage import transform
import glob
from tqdm import notebook, tqdm
from skimage.transform import rotate
from sklearn.model_selection import train_test_split


def squeeze_collate_fn(batch):
    item = [each for each in batch if each is not None]
    if item:
        item = item[0]
        return ((item.image.squeeze(), item.target.unsqueeze(0)))
    return []


class BCELossMixin(object):
       
    
    def loss(self, prediction, target, weights=True):
        loss = nn.BCEWithLogitsLoss()
        return loss(prediction, target)
    
    def _weighted_loss(self, weights=None):
        pass
    
    

class VerifyDicomFilesMixin(object):
    
    def is_valid(self, item):
        
        img = Path(str(item.get("image")))
        msk = Path(str(item.get("mask")))
        
        if not all([img.exists(), msk.exists()]):
            return False
        
        if not all([len(os.listdir(img)), len(os.listdir(msk))]):
            return False
        
        return True
    
    def clean(self):
        clean_items = []
        for item in notebook.tqdm(self.items, desc="Verifying Data"):
            if self.is_valid(item):
                clean_items.append(item)
        self.items = clean_items
        
        
    
        

class MRI3dDataSet(BCELossMixin, VerifyDicomFilesMixin):
    
    def __init__(self, items, transform=None):
        self.items = items       
        self.transform = transform or (lambda item: item)
        
        if hasattr(self, "clean"):
            self.clean()
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        try:
            item = self.items[idx]
            item = self.transform(self.items[idx])
            return item
        except:
            return None
        
        
