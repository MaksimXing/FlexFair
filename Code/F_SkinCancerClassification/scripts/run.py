import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.utils.data
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code.F_SkinCancerClassification.method import *
import numpy as np
import random
import argparse

datapath = # skin data path here

parser = argparse.ArgumentParser(description='PyTorch Sketch Me That Shoe Example')

parser.add_argument('--net', type=str, default='resnet50', help='The model to be used (vgg16, resnet34, resnet50, resnet101, resnet152)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>,<save_latest_freq>+<epoch_count>...')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Adm weight decay')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--classes', type=int, default=400,
                    help='number of classes')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='NetModel', type=str,
                    help='name of experiment')
parser.add_argument('--lr_decay_iters', default=5)

parser.add_argument('--com_round', default=1)
parser.add_argument('--miu', default=0.001)
parser.add_argument('--rho', default=0.9)
parser.add_argument('--c_lr', default=0.9)

parser.add_argument('--fairness_step', default=70)
parser.add_argument('--penalty_weight', default=2.0)
parser.add_argument('--k', default=30)
parser.add_argument('--dp_eo', default='dp', help='dp / eo / noweight')
parser.add_argument('--sex_age', default='age')
parser.add_argument('--method', default='flexfair')
parser.add_argument('--datapath', default=datapath)

parser.add_argument('--target_GPU', type=str, default='1', help='Choose which GPU to use')
parser.add_argument('--compute_type', type=str, default='acc', help='acc / ap')

# fairfed: beta
parser.add_argument('--beta', type=float, default=0.01)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    penalty = args.penalty_weight
    os.environ['CUDA_VISIBLE_DEVICES'] = args.target_GPU
    
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.method == 'flexfair':
        if args.sex_age == 'sex':
            NewWeightedSexTrain(args, kwargs)
        elif args.sex_age == 'age':
            NewWeightedAgeTrain(args, kwargs)
    elif args.method == 'fedavg':
        FedAvgTrain(args, kwargs)
    elif args.method == 'fedprox':
        FedProxTrain(args, kwargs)
    elif args.method == 'fednova':
        FedNovaTrain(args, kwargs)
    elif args.method == 'scaffold':
        ScaffoldTrain(args, kwargs)
    elif args.method == 'fairmixup':
        if args.sex_age == 'sex':
            FairMixupSexTrain(args, kwargs)
        elif args.sex_age == 'age':
            FairMixupAgeTrain(args, kwargs)
    elif args.method == 'fairfed':
        if args.sex_age == 'sex':
            FairFedSexTrain(args, kwargs)
        elif args.sex_age == 'age':
            FairFedAgeTrain(args, kwargs)