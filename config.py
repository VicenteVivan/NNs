import argparse
import torch
import multiprocessing

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.kernels = 10

    opt.resources = "./"

    opt.size = 224

    opt.n_epochs = 150

    opt.description = '100M'
    opt.archname = 'Sequential NN'
    opt.evaluate = False

    opt.lr = 1e-2
    opt.step_size = 3

    opt.batch_size = 32
    opt.trainset = 'SD2_train'
    opt.testset = 'SD2_test'
    opt.device = torch.device('cuda')

    return opt
