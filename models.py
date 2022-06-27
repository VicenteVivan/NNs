from matplotlib.pyplot import get
import torch
import torch.nn as nn

def getNetwork(input_size, output_size, num_hidden_layers, hidden_layer_size):
    network = nn.Sequential()
    network.add_module("input", nn.Linear(input_size, hidden_layer_size))
    network.add_module("relu", nn.LeakyReLU())
    for i in range(num_hidden_layers - 1):
        network.add_module("dropout" + str(i), nn.Dropout(0.5))
        network.add_module("hidden" + str(i), nn.Linear(hidden_layer_size, hidden_layer_size))
        network.add_module("relu" + str(i), nn.LeakyReLU())
    network.add_module("dropout", nn.Dropout(0.5))
    network.add_module("output", nn.Linear(hidden_layer_size, output_size))
    network.add_module("tanh", nn.Tanh())
    return network

name1 = "(i = 1): 196"
net1 = getNetwork(50, 1, 1, 1961)

name2 = "(i = 2): 78"
net2 = getNetwork(50, 1, 2, 292)

name3 = "(i = 3): 59"
net3 = getNetwork(50, 1, 3, 211)

name4 = "(i = 4): 50"
net4 = getNetwork(50, 1, 4, 174)

name5 = "(i = 5): 44"
net5 = getNetwork(50, 1, 5, 152)

name6 = "(i = 6): 40"
net6 = getNetwork(50, 1, 6, 136)

name7="(i = 7): 37"
net7 = getNetwork(50, 1, 7, 125)

name8="(i = 8): 34"
net8 = getNetwork(50, 1, 8, 116)

name9="(i = 9): 32"
net9 = getNetwork(50, 1, 9, 109)

name10="(i = 10): 31"
net10 = getNetwork(50, 1, 10, 103)

def getModels():
    return [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10]

def getNames():
    return [name1, name2, name3, name4, name5, name6, name7, name8, name9, name10]

if __name__ == "__main__":
    print(getModels())
    print(getNames())

