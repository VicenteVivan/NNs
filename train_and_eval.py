from sklearn.metrics import accuracy_score
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm

import wandb
import pandas as pd
import dataloader

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),])

from config import getopt
from models import getModels, getNames

def getTrainingAccuracy(y_pred, y_true, opt=None): 
    # Discretize the predictions
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    # y_pred = np.where(y_pred > 0, 1, 0)
    # y_pred = np.where(y_pred == 0, -1, y_pred)
    
    y_pred = np.argmax(y_pred, axis=1)

    # Save the predictions
    acc = accuracy_score(y_true, y_pred)
    
    return acc

def train(train_dataloader, model, model_name, criterion, optimizer, opt, epoch, val_dataloader=None):
    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0

    val_cycle = 100
    print("Outputting loss every", val_cycle, "batches")
    print("Validating every", val_cycle*5, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))

    for i ,(X, y) in bar:
        batch_size = X.shape[0]

        X = X.to(opt.device)
        y = y.to(opt.device)
    
        optimizer.zero_grad()

        preds = model(X)
        
        # Convert y (vector with indices of the correct class) to one-hot matrix
        # y_onehot = torch.zeros(preds.shape[0], preds.shape[1]).to(opt.device)
        # y_onehot.scatter_(1, y.unsqueeze(1), 1)
        # y_onehot = y_onehot.to(opt.device)
        
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        
        if i % val_cycle == 0:
            wandb.log({"Training Loss" : {model_name: loss.item()}})
            wandb.log({"Training Accuracy" : {model_name: getTrainingAccuracy(preds, y, opt)}})
            bar.set_description("Epoch {} Loss: {:.4f}".format(epoch, epoch_loss))

        if val_dataloader != None and i % (val_cycle * 5) == 0:
            evaluate(val_dataloader, model, model_name, criterion, epoch, opt)
    
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    return np.mean(losses)

def evaluate(val_dataloader, model, model_name, criterion, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    loss = 0.0
    preds = []
    targets = []
    
    model.eval() 
    
    for i, (X, y) in bar:
        y = y.to(opt.device)
        X = X.to(opt.device)
        
        with torch.no_grad():
            y_pred = model(X)
            
        # Convert y (vector with indices of the correct class) to one-hot matrix
        # y_onehot = torch.zeros(y_pred.shape[0], y_pred.shape[1]).to(opt.device)
        # y_onehot.scatter_(1, y.unsqueeze(1), 1)
        # y_onehot = y_onehot.to(opt.device)
            
        loss += criterion(y_pred)
        
        # Discretize the predictions
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
        y_pred = np.argmax(y_pred, axis=1)
        
        # Save the predictions
        targets.append(y)
        preds.append(y_pred)
        
    model.train() 

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    loss /= len(val_dataloader)
    wandb.log({"Validation Loss" : {model_name: loss.item()}})
    
    print(targets.shape, preds.shape)
    print(type(targets), type(preds))
    acc = accuracy_score(targets, preds)
    print("Accuracy is", acc)
    wandb.log({"Validation Accuracy" : {model_name: acc}})

if __name__ == '__main__':
    opt = getopt()
    
    w = wandb.init(project='NNs',
                    entity='vicentevivan',
                    group="MNIST Switch")
    
    model = getModels()[0]
    model_name = getNames()[0]
    # model = model.to(opt.device)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
