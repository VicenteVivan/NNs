from binascii import a2b_hex
from unicodedata import name
from matplotlib.pyplot import get
import torch
import torch.nn as nn
import numpy as np
import math

# Convolutional LeNet CIFAR10 (2 Convolutions + Subsampling) Goes to 120 Fully Connected Layers Input size

a = 512
c = 100
P = 1_000_000

# input_size = 1568
input_size = 2048

def f(i):
    if (i == 1):
        return P / (a + c)
    else:
        return (-(a+c) + math.sqrt((a+c)**2 + 4*P*(i - 1))) / (2*(i- 1))

f = np.vectorize(f)

L = np.array([(0 + 1), (2 + 1), (4 + 1), (8 + 1), (16 + 1), (32 + 1), (64 + 1), (128 + 1)])
L = np.array([int(np.round(f(i))) for i in L])

class ResMLPBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 self.batch_norm1,
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size))
        
    def forward(self, x):
        x = self.mlp(x) + x
        x = self.batch_norm2(x)
        x = nn.ReLU()(x)
        return x

def getNetwork(input_size, output_size, num_hidden_layers, hidden_layer_size):
    network = nn.Sequential()
    
    # Convolutional Neural Network 
    network.add_module("conv1", nn.Conv2d(in_channels=3,
                                            out_channels=16,
                                            kernel_size=5,
                                            stride=1,
                                            padding=2))         
    network.add_module("relu1", nn.ReLU())
    network.add_module("maxpool1", nn.MaxPool2d(kernel_size=2))
    network.add_module("conv2", nn.Conv2d(16, 32, 5, 1, 2))
    network.add_module("relu2", nn.ReLU())
    network.add_module("maxpool2", nn.MaxPool2d(2))
    
    # Flatten 32 * 7 * 7
    network.add_module("flatten", nn.Flatten())
    network.add_module("linear", nn.Linear(input_size, a))

    # MLP Head
    network.add_module("input", nn.Linear(in_features = a, out_features=hidden_layer_size))
    network.add_module("relu", nn.ReLU())
    for i in range(int((num_hidden_layers - 1) / 2)):
        network.add_module("hidden" + str(i), ResMLPBlock(hidden_layer_size))
    network.add_module("output", nn.Linear(hidden_layer_size, output_size))
    return network

# Nets 1-20

name1 = f'(i = 1): {L[0]}'
net1 = getNetwork(input_size, c, 1, L[0])

name2 = f'(i = 2): {L[1]}'
net2 = getNetwork(input_size, c, 3, L[1])

name3 = f'(i = 4): {L[2]}'
net3 = getNetwork(input_size, c, 5, L[2])

name4 = f'(i = 8): {L[3]}'
net4 = getNetwork(input_size, c, 9, L[3])

name5 = f'(i = 16): {L[4]}'
net5 = getNetwork(input_size, c, 17, L[4])

name6 = f'(i = 32): {L[5]}'
net6 = getNetwork(input_size, c, 33, L[5])

name7 = f'(i = 64): {L[6]}'
net7 = getNetwork(input_size, c, 65, L[6])

name8 = f'(i = 128): {L[7]}'
net8 = getNetwork(input_size, c, 129, L[7])

name9 = f'(i = 256): {L[8]}'
net9 = getNetwork(input_size, c, 257, L[8])

def getModels():
    return [net9, net8, net7, net6, net5, net4, net3, net2, net1]
    #return [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10]


def getNames():
    return [name9, name8, name7, name6, name5, name4, name3, name2, name1]
    #return [name1, name2, name3, name4, name5, name6, name7, name8, name9, name10]

if __name__ == "__main__":
    models = getModels()
    print(getModels())
    print(getNames())
    
    # Test Model 1
    X = torch.rand((1,3,32,32))
    
    net = models[0]
    
    net.forward(X)
    

