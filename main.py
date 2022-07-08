if __name__ == '__main__':
    import os, numpy as np, argparse, time
    from tqdm import tqdm

    import torch
    import torch.nn as nn

    import dataloader
    from train_and_eval import train, evaluate
    
    import wandb

    import models 
    from config import getopt
    
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    opt = getopt()

    config = {
        'learning_rate' : opt.lr,
        'epochs' : opt.n_epochs,
        'batch_size' : opt.batch_size,
        'architecture' : opt.archname
        }

    # SD2 Dataset
    # train_dataset = dataloader.SD2(split='train', opt=opt)
    # val_dataset = dataloader.SD2(split='test', opt=opt)

    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

    #criterion = torch.nn.MSELoss()
    
    # Load MNIST
    # train_dataset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    # val_dataset= datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    # Load FashionMNIST
    train_dataset = datasets.FashionMNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    val_dataset= datasets.FashionMNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    # Load CIFAR10
    # train_dataset = datasets.CIFAR10('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    # val_dataset= datasets.CIFAR10('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()

    NN_Models = models.getModels()
    NN_Names = models.getNames()

    for i, (model, model_name) in enumerate(zip(NN_Models, NN_Names)):
        w = wandb.init(project='F-MNIST',
                       entity='vicentevivan',
                       reinit=True,
                       config=config)
        
        wandb.run.name = opt.description
        
        model = model.to(opt.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        _ = model.to(opt.device)

        wandb.watch(model, criterion, log="all")

        for epoch in range(opt.n_epochs):
            evaluate(val_dataloader=val_dataloader, model=model, model_name=model_name, criterion=criterion, epoch=epoch, opt=opt)
            if not opt.evaluate:
                _ = model.train()

                loss = train(train_dataloader=train_dataloader, model=model, model_name=model_name, criterion=criterion, optimizer=optimizer, opt=opt, epoch=epoch)
            
        
        w.finish()
