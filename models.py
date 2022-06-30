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

name1 = "(i = 1): 324"
net1 = getNetwork(3072, 10, 1, 324)

name2 = "(i = 2): 296"
net2 = getNetwork(3072, 10, 2, 296)

name3 = "(i = 3): 275"
net3 = getNetwork(3072, 10, 3, 275)

name4 = "(i = 4): 259"
net4 = getNetwork(3072, 10, 4, 259)

name5 = "(i = 5): 246"
net5 = getNetwork(3072, 10, 5, 246)

name6 = "(i = 6): 235"
net6 = getNetwork(3072, 10, 6, 235)

name7="(i = 7): 225"
net7 = getNetwork(3072, 10, 7, 225)

name8="(i = 8): 217"
net8 = getNetwork(3072, 10, 8, 217)

name9="(i = 9): 210"
net9 = getNetwork(3072, 10, 9, 210)

name10="(i = 10): 204"
net10 = getNetwork(3072, 10, 10, 204)

def getModels():
    return [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10]

def getNames():
    return [name1, name2, name3, name4, name5, name6, name7, name8, name9, name10]

if __name__ == "__main__":
    print(getModels())
    print(getNames())

