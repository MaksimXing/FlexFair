from Code.F_FundusVascularSeg.method import *
import random
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# output_path = '../output/'

if __name__ == '__main__':
    seed = int(sys.argv[1])
    method = sys.argv[2]
    weight = float(sys.argv[3])
    gpu = sys.argv[4]
    
    output_path = f'../output/{method}/s{seed}w{weight}/'

    class Args:
        def __init__(self):
            ## set the backbone type
            self.backbone = 'res2net50'
            ## set the path of training dataset
            self.datapaths = ['../dataset/CHASEDB1', '../dataset/DRIVE', '../dataset/STARE']
            self.datapath = ''
            self.dataset = ['CHASEDB1', 'DRIVE', 'STARE']
            ## set the path of logging
            self.output_path = output_path
            ## keep unchanged
            self.mode = 'train'
            self.epoch = 46
            # self.epoch = 3
            self.batch_size = 4
            self.lr = 0.001
            self.num_workers = 4
            self.weight_decay = 1e-3
            self.clip = 0.5
            self.com_round = 1
            self.miu = 0.001
            self.rho = 0.9
            self.fairness_step = 5
            self.seed = seed
            self.use_swa = True
            self.swa_start = 0.75
            self.rho_sam = 0.05
            self.eta_sam = 0.01
            
            if method == 'FedNova':
                self.rho = weight
            elif method == 'FedProx':
                self.miu = weight
            elif method == 'Scaffold':
                self.c_lr = weight
            elif method == 'FedAvgOOD':
                self.penalty_weight = weight
            elif method == 'FairMixup':
                self.penalty_weight = weight
            elif method == 'FairFed':
                self.beta = weight
                

    # FIXED RANDOM SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    ## training
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if method == 'FedAvg':
        FedAvgTrain(Args()).forward()
    elif method == 'FedNova':
        FedNovaTrain(Args()).forward()
    elif method == 'FedProx':
        FedProxTrain(Args()).forward()
    elif method == 'Scaffold':
        ScaffoldTrain(Args()).forward()
    elif method == 'FedAvgOOD':
        FedAvgOODTrain(Args()).forward()
    elif method == 'FairMixup':
        FairMixupTrain(Args()).forward()
    elif method == 'FairFed':
        FairFedTrain(Args()).forward()