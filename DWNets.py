# Import libraries  
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time

def getDataPoint2():
    """Creates a random data point for the second synthetic dataset
    
    Returns:
        tuple: (x (np.array), y (float))
    """
    # Randomly sample label from {-1, 1}
    y = np.random.choice([-1, 1], p=[0.5, 0.5])
    
    # Create data point & coordinates
    x = np.random.uniform(-1, 1, 50)
    
    # Set first coordinate
    if (y == 1):
        x[0] = random.normalvariate(0, 1) 
    else:
        x[0] = random.normalvariate(2, 1)
        
    return x, y
        
        
# Create dataset 
SD2 = pd.DataFrame(columns=[f'x{i}' for i in range(50)] + ['y'])

# Add data points
for i in range(10000):
    # Get data point
    x, y = getDataPoint2()
    
    # Add data point to dataset
    SD2.loc[i] = np.append(x, y)
    
    if (i % 1000 == 0):
        print(f'{i/1000}%')
    
# Split dataset in 2 pandas dataframes: train and test
SD2_train, SD2_test = train_test_split(SD2, test_size=0.2, random_state=42)

# Save dataset as csv
SD2_train.to_csv('SD2_train.csv', index=False)
SD2_test.to_csv('SD2_test.csv', index=False)