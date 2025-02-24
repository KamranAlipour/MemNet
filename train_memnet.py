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
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.1, help="the initial learning rate is 0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate is divided 10 every 20 epochs")
parser.add_argument("--cuda", default=True,action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="checkpoint1/model_epoch_38.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpu ids (default: 0,1,2,3)")

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromNpys("data/npy_frames_split_1024_train")
    #training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = MemNet(1, 64, 6, 6)
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        #model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()  #multi-card data parallel
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every num(opt.step) epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        input = input[:,:,:,0].unsqueeze(1)
        target = target[:,:,:,0].unsqueeze(1)
        #loss = criterion(model(input), target)
        pdb.set_trace()
        prediction = model(input)
        loss = criterion(prediction, target)
        print("Outside: input size", input.size(),"prediction_size", prediction.size())
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint1/" + "model_epoch_{}.pth".format(epoch)
    #state = {"epoch": epoch ,"model": model.state_dict}
    state = {"epoch": epoch ,"model": model.state_dict()}
    if not os.path.exists("checkpoint1/"):
        os.makedirs("checkpoint1/")

    torch.save(state, model_out_path)  # save weights and network architecture

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
