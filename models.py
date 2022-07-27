from matplotlib.pyplot import get
import torch
import torch.nn as nn
import numpy as np
import math

# Convolutional LeNet CIFAR10 (2 Convolutions + Subsampling) Goes to 120 Fully Connected Layers Input size

a = 2048
c = 100
P = 100_000_000

# input_size = 1568
input_size = 2048

def f(i):
    if (i == 1):
        return P / (a + c)
    else:
        return (-(a+c) + math.sqrt((a+c)**2 + 4*P*(i - 1))) / (2*(i- 1))
    
f = np.vectorize(f)

L = np.array([f(i) for i in range(1, 11)])
L = np.array([int(np.round(f(i))) for i in range(1, 11)])

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

    # MLP Head
    network.add_module("input", nn.Linear(in_features=input_size, out_features=hidden_layer_size))
    network.add_module("relu", nn.ReLU())
    for i in range(num_hidden_layers - 1):
        network.add_module("hidden" + str(i), nn.Linear(hidden_layer_size, hidden_layer_size))
        network.add_module("relu" + str(i), nn.ReLU())
    network.add_module("output", nn.Linear(hidden_layer_size, output_size))
    return network

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

def getModels():
    return [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10]

def getNames():
    return [name1, name2, name3, name4, name5, name6, name7, name8, name9, name10]

if __name__ == "__main__":
    models = getModels()
    print(getModels())
    print(getNames())
    
    # Test Model 1
    X = torch.rand((1,3,32,32))
    
    net = models[0]
    
    net.forward(X)
    

