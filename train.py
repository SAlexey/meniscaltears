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


def train(model, loaders, optimizer, epochs=30):
    
    root_dir = Path("models")
    
    
    # BOOK KEEPING
    # save model architechture and create directories for model outputs
    
    if not root_dir.exists():
        root_dir.mkdir()
        
    model_dir = root_dir / model.__class__.__name__
    
    if not model_dir.exists():
        model_dir.mkdir()
    
    with open(model_dir / "model.txt", "w") as model_description:
        model_description.write(f"EPOCHS: {epochs}\n")
        model_description.write(str(model))
        model_description.write(str(optimizer))
    # END BOOK KEEPING
    
    

    best_loss = np.inf
    
    validation_score = []
    training_score = []
    
    validation_loss = []
    training_loss = []
    
    threshold = 0.5

    
    # training loop
    for epoch in range(epochs):
        
        print(f"EPOCH {epoch + 1}")
        print("=" * 60)
    
        model.train()

        t_loss, predictions, labels = run(model, loaders["training"], optimizer=optimizer, train=True)
        labels = labels.astype(int)[0]
 
        training_loss.append(t_loss)        
        avg_train_loss = t_loss.mean()
        
        thresholded = (predictions >= threshold).astype(int)[0]
        bacc_score = metrics.balanced_accuracy_score(labels, thresholded)
                
       
        print(f"LOSS {avg_train_loss:.4f} - ACCURACY {bacc_score:.3f}")
        
        cm = metrics.confusion_matrix(labels, thresholded)
        disp = metrics._plot.confusion_matrix.ConfusionMatrixDisplay(cm, ["normal", "sick"])
        disp.plot()
        plt.show()
        
        
        training_score.append(bacc_score)
        
        model.eval()

        v_loss, predictions, labels = run(model, loaders["validation"])
        labels = labels.astype(int)[0]
        
        validation_loss.append(v_loss)
        avg_val_loss = v_loss.mean()
        
        thresholded = (predictions >= threshold).astype(int)[0]
        bacc_score = metrics.balanced_accuracy_score(labels, thresholded)
        
        print(f"LOSS {avg_val_loss:.4f} - ACCURACY {bacc_score:.3f}")
        
        cm = metrics.confusion_matrix(labels, thresholded)
        disp = metrics._plot.confusion_matrix.ConfusionMatrixDisplay(cm, ["normal", "sick"])
        disp.plot()
        plt.show()
        
        validation_score.append(bacc_score)

        if avg_val_loss < best_loss:

            # save model params
            best_loss = avg_val_loss
            model_name = f"train{avg_train_loss:.4f}_val{avg_val_loss:.4f}_epoch{epoch}"

            torch.save(model.state_dict(), model_dir / f"{model_name}.state_dict")
            
            
    np.save(model_dir / "validation_score.npy", validation_score)
    np.save(model_dir / "training_score.npy", training_score)
    np.save(model_dir / "training_loss.npy", np.hstack(training_loss))
    np.save(model_dir / "validation_loss.npy", np.hstack(validation_loss))


def run(model, loader, optimizer=None, train=False, writer=None):
    
    _predictions, _labels, _losses = [], [], []
    
    desc = "training" if train else "validation"
            
    for batch in notebook.tqdm(loader, desc=desc):
        
        # batch is a tuple (image, label)
        # image is a float tensor of shape (slices, channels, height, width)
        # example 3D MRI with dimensions (50, 250, 250)
        # will be transformed during loading into (50, 3, 250, 250)
        
        if not batch:
            continue
        
        inputs, labels = batch
        
        if train:
            optimizer.zero_grad()
        
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()
        
        outputs = model(inputs)
        
        predictions = torch.sigmoid(outputs)
        
        prediction_npy = predictions.detach().cpu().numpy()
        label_npy = labels.detach().cpu().numpy()
        
        loss = loader.dataset.loss(outputs, labels)
        loss_npy = loss.detach().cpu().numpy()
        
        _predictions.append(prediction_npy)
        _labels.append(label_npy)
        _losses.append(loss_npy)
        
        
        if train:
            loss.backward(retain_graph=True)
            optimizer.step()
            
    return np.hstack(_losses), np.hstack(_predictions), np.hstack(_labels)


def plot_model_output(model, epochs=None, scoring="score"):
    model_name = model.__class__.__name__
    model_dir  = Path("models") / model_name 
    
    validation_loss = np.load(model_dir / "validation_loss.npy")
    training_loss = np.load(model_dir / "training_loss.npy")
    
    validation_score = np.load(model_dir / "validation_score.npy")
    training_score = np.load(model_dir / "training_score.npy")
    
    if epochs is not None:
        validation_loss = np.array_split(validation_loss, epochs)
        validation_loss = list(map(np.mean, validation_loss))
        
        training_loss = np.array_split(training_loss, epochs)
        training_loss = list(map(np.mean, training_loss))
        
        validation_score = np.array_split(validation_score, epochs)
        validation_score = list(map(np.mean, validation_score))
        
        training_score = np.array_split(training_score, epochs)
        training_score = list(map(np.mean, training_score))
        
    pd.DataFrame({
        "training": savgol_filter(training_loss, 5, 3), 
        "validation": savgol_filter(validation_loss, 5, 3),
    }).plot()
    plt.ylabel("Binary Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(model_dir / "BCELoss.png", bbox_inches="tight")
    
    pd.DataFrame({
        "training": savgol_filter(training_score, 5, 3), 
        "validation": savgol_filter(validation_score, 5, 3),
    }).plot()
    plt.ylabel(scoring)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(model_dir / "Scoring.png", bbox_inches="tight")