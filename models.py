from matplotlib.pyplot import get
import torch
import torch.nn as nn

# Convolutional LeNet CIFAR10 (2 Convolutions + Subsampling) Goes to 120 Fully Connected Layers Input size


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
    network.add_module("input", nn.Linear(in_features=2048, out_features=hidden_layer_size))
    network.add_module("relu", nn.ReLU())
    for i in range(num_hidden_layers - 1):
        network.add_module("hidden" + str(i), nn.Linear(hidden_layer_size, hidden_layer_size))
        network.add_module("relu" + str(i), nn.ReLU())
    network.add_module("output", nn.Linear(hidden_layer_size, output_size))
    network.add_module("softmax", nn.Softmax(dim=1))
    return network

# input_size = 1568
input_size = 2048

name1 = "(i = 1): 46555"
net1 = getNetwork(input_size, 10, 1, 46555)

name2 = "(i = 2): 8984"
net2 = getNetwork(input_size, 10, 2, 8984)

name3 = "(i = 3): 6554"
net3 = getNetwork(input_size, 10, 3, 6554)

name4 = "(i = 4): 5427"
net4 = getNetwork(input_size, 10, 4, 5427)

name5 = "(i = 5): 4739"
net5 = getNetwork(input_size, 10, 5, 4739)

name6 = "(i = 6): 4262"
net6 = getNetwork(input_size, 10, 6, 4262)

name7="(i = 7): 3907"
net7 = getNetwork(input_size, 10, 7, 3907)

name8="(i = 8): 3629"
net8 = getNetwork(input_size, 10, 8, 3629)

name9="(i = 9): 3404"
net9 = getNetwork(input_size, 10, 9, 3404)

name10="(i = 10): 3216"
net10 = getNetwork(input_size, 10, 10, 3216)

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
    

