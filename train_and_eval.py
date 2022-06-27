from sklearn.metrics import accuracy_score
import numpy as np

import torch
import torch.nn.functional as F
import pickle

from tqdm import tqdm

import wandb
import pandas as pd
import json
import dataloader

from config import getopt
from models import getModels, getNames

def getTrainingAccuracy(y_pred, y_true, opt=None):
    y_pred.to(opt.device)
    y_true.to(opt.device)
    
    # Discretize the predictions
    y_pred = torch.where(y_pred > 0, torch.ones(y_pred.shape).to(opt.device), torch.zeros(y_pred.shape).to(opt.device)).to(opt.device)
    y_pred = torch.where(y_pred == 0, torch.ones(y_pred.shape).to(opt.device) * -1, y_pred).to(opt.device)
    
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

    for i, (X, y) in bar:
        y = y.to(opt.device)
        X = X.to(opt.device)
        
        with torch.no_grad():
            y_pred = model(X)
            
        loss += criterion(y_pred, y)
        
        # Discretize the predictions
        y_pred = torch.where(y_pred > 0, torch.ones(y_pred.shape).to(opt.device), torch.zeros(y_pred.shape).to(opt.device)).to(opt.device)
        y_pred = torch.where(y_pred == 0, torch.ones(y_pred.shape).to(opt.device) * -1, y_pred).to(opt.device)
        
        # Save the predictions
        targets.append(y)
        preds.append(y_pred)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    loss /= len(val_dataloader)
    wandb.log({"Validation Loss" : {model_name: loss.item()}})
    
    acc = accuracy_score(targets, preds)
    print("Accuracy is", acc)
    wandb.log({"Validation Accuracy" : {model_name: acc}})

if __name__ == '__main__':
    opt = getopt()
    
    model = getModels()[0]
    model_name = getNames()[0]
    model = model.to(opt.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    train_dataset = dataloader.SD2(split='train', opt=opt)
    val_dataset = dataloader.SD2(split='test', opt=opt)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
    
    for epoch in range(3):
        if not opt.evaluate:
            _ = model.train()

            loss = train(train_dataloader=train_dataloader, model=model, model_name=model_name, criterion=criterion, optimizer=optimizer, opt=opt, epoch=epoch)

        evaluate(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)
    
    
