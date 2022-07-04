import argparse
import torch
import multiprocessing

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    # opt.kernels = multiprocessing.cpu_count()
    opt.kernels = 4

    opt.resources = "./"

    opt.size = 224

    opt.n_epochs = 50

    opt.description = '100K P, 10K S'
    opt.archname = 'Sequential NN'
    opt.evaluate = False

    opt.lr = 5e-4
    opt.step_size = 3

    opt.batch_size = 32
    opt.trainset = 'SD2_train'
    opt.testset = 'SD2_test'
    opt.device = torch.device('cpu')

    return opt
