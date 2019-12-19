######################################################3
#  
#  An implementation of Memnet,ICCV 2017
#  Author: Rosun
#  Data: 2018/3/31
#
#######################################################
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from memnet1 import MemNet
from dataset import DatasetFromHdf5, DatasetFromNpys

import pdb
# Training settings
parser = argparse.ArgumentParser(description="PyTorch MemNet")
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.1, help="the initial learning rate is 0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate is divided 10 every 20 epochs")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="checkpoint1/model_epoch_38.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpu ids (default: 0,1,2,3)")

global opt
opt = parser.parse_args()

train_set = DatasetFromNpys("data/npy_frames_split_1024_train")
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)

for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        pdb.set_trace()
