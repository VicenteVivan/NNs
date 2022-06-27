import argparse
import multiprocessing
import torch

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = multiprocessing.cpu_count()

    opt.resources = "./"

    opt.size = 224

    opt.n_epochs = 100

    opt.description = '1M Params, 10K Samples'
    opt.archname = 'Sequential NN'
    opt.evaluate = False

    opt.lr = 5e-4
    opt.step_size = 3

    opt.batch_size = 256
    opt.trainset = 'SD2_train'
    opt.testset = 'SD2_test'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return opt
