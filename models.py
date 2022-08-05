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

L = np.array([f(i) for i in range(1, 16)])
L = np.array([int(np.round(f(i))) for i in range(1, 16)])

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
    for i in range(num_hidden_layers - 1):
        network.add_module("hidden" + str(i), nn.Linear(hidden_layer_size, hidden_layer_size))
        network.add_module("batchnorm" + str(i), nn.BatchNorm1d(hidden_layer_size))
        network.add_module("relu" + str(i), nn.ReLU())
    network.add_module("output", nn.Linear(hidden_layer_size, output_size))
    network.add_module("softmax", nn.Softmax(dim=1))
    return network

# Nets 1-20

name1 = f'(i = 1): {L[0]}'
net1 = getNetwork(input_size, c, 1, L[0])

name2 = f'(i = 2): {L[1]}'
net2 = getNetwork(input_size, c, 2, L[1])

name3 = f'(i = 3): {L[2]}'
net3 = getNetwork(input_size, c, 3, L[2])

name4 = f'(i = 4): {L[3]}'
net4 = getNetwork(input_size, c, 4, L[3])

name5 = f'(i = 5): {L[4]}'
net5 = getNetwork(input_size, c, 5, L[4])

name6 = f'(i = 6): {L[5]}'
net6 = getNetwork(input_size, c, 6, L[5])

name7 = f'(i = 7): {L[6]}'
net7 = getNetwork(input_size, c, 7, L[6])

name8 = f'(i = 8): {L[7]}'
net8 = getNetwork(input_size, c, 8, L[7])

name9 = f'(i = 9): {L[8]}'
net9 = getNetwork(input_size, c, 9, L[8])

name10 = f'(i = 10): {L[9]}'
net10 = getNetwork(input_size, c, 10, L[9])

name11 = f'(i = 11): {L[10]}'
net11 = getNetwork(input_size, c, 11, L[10])

name12 = f'(i = 12): {L[11]}'
net12 = getNetwork(input_size, c, 12, L[11])

name13 = f'(i = 13): {L[12]}'
net13 = getNetwork(input_size, c, 13, L[12])

name14 = f'(i = 14): {L[13]}'
net14 = getNetwork(input_size, c, 14, L[13])

name15 = f'(i = 15): {L[14]}'
net15 = getNetwork(input_size, c, 15, L[14])

def getModels():
    return [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12, net13, net14, net15]

def getNames():
    return [name1, name2, name3, name4, name5, name6, name7, name8, name9, name10, name11, name12, name13, name14, name15]

if __name__ == "__main__":
    models = getModels()
    print(getModels())
    print(getNames())
    
    # Test Model 1
    X = torch.rand((1,3,32,32))
    
    net = models[0]
    
    net.forward(X)
    

