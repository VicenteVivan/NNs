import csv

from torch.utils.data import Dataset

import os
import torch

import pandas as pd

import numpy as np
import glob
import random

import torchvision.transforms as transforms 
from torchvision.utils import save_image

import csv

import json
from collections import Counter

import matplotlib.pyplot as plt
from os.path import exists

from config import getopt

class SD2(Dataset):

    def __init__(self, split='train', opt=None):

        np.random.seed(0)
        
        self.split = split 
        
        if split == 'train':
            xy = pd.read_csv(opt.resources + 'SD2_train.csv')
        if split == 'test':
            xy = pd.read_csv(opt.resources + 'SD2_test.csv')
            
        # Get X and y
        self.X = xy.iloc[:,:-1]
        self.y = xy.iloc[:,-1]

        # Convert DataFrame to Torch Tensor
        self.X = torch.from_numpy(self.X.values).float()
        self.y = torch.from_numpy(self.y.values).float().reshape(-1,1)
        
        self.data = self.X
        
        print("Loaded data, total number of samples:", len(self.X))

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    opt = getopt()

    dataset = SD2(split='train', opt=opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)

    for i, (X, y) in enumerate(dataloader):
        print(X.shape, y.shape)
        print(X)
        print(y)
        break
