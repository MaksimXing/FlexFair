import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from numpy.random import beta

sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
from utils import TrainData, TestData, mkdir, evaluateGlobal
from network import clip_gradient
import matplotlib.pyplot as plt
import csv

plt.ion()


class FairMixupTrain:
    def __init__(self, args):
        # init args
        self.args = args

        # init output
        now = datetime.now()
        local_time = now.strftime("%m%d-%H%M%S")
        self.path = args.output_path + local_time
        mkdir(self.path)

        # init output csv
        self.logger_list = []
        for index in range(0, 3):
            self.logger_list.append(SummaryWriter(self.path + '/log/' + args.dataset[index]))
        f = open(self.path + '/result.csv', "w", newline="")
        self.writer = csv.writer(f)


        ## init model
        self.model_list = []
        for index in range(0, 3):
            self.model_list.append(Model(args).cuda())
            self.model_list[index].train()

        # init global model
        self.global_model = Model(args).cuda()
        self.global_model.train()

        model_state_dict = torch.load(f'../pretrain/avg{args.seed}_model.pth')
        self.global_model.load_state_dict(model_state_dict, strict = True)
        
        ## init dataset
        self.dataset_list = []
        self.loader_list = []
        for index in range(0, 3):
            args.datapath = args.datapaths[index]
            self.dataset_list.append(TrainData(args))
            self.loader_list.append(DataLoader(dataset=self.dataset_list[index], batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True))
        args.datapath = ''

        ## init opt
        self.max_dice_list = [0, 0, 0]
        self.max_dice_global = 0
        self.optimizer_list = []
        self.lr_scheduler_list = []
        for index in range(0, 3):
            self.optimizer_list.append(torch.optim.AdamW(self.model_list[index].parameters(), lr=args.lr))
            self.lr_scheduler_list.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_list[index], T_max=args.epoch, eta_min=0))

    def forward(self):
        global_step_list = [0, 0, 0]
        com_counter = 0
        result = -1

        # get dataset size
        dataset_size = []
        dataset_total_size = 0
        for loader in self.loader_list:
            dataset_size.append(len(loader.dataset))
            dataset_total_size += len(loader.dataset)

        for epoch in range(self.args.epoch):
            # init client with global
            global_para = self.global_model.state_dict()
            # Init
            if com_counter == 0:
                # replace client with global(for init)
                for index in range(0, 3):
                    self.model_list[index].load_state_dict(global_para)
            # detect fed
            elif com_counter % self.args.com_round == 0:
                print("*** FairMixup Execute ***")
                for index in range(0, 3):
                    sub_net_para = self.model_list[index].state_dict()
                    # FedAvg
                    if index == 0:
                        for key in sub_net_para:
                            global_para[key] = sub_net_para[key] \
                                               * (dataset_size[index] * 1.0 / dataset_total_size)
                    else:
                        for key in sub_net_para:
                            global_para[key] += sub_net_para[key] \
                                                * (dataset_size[index] * 1.0 / dataset_total_size)
                # update global
                self.global_model.load_state_dict(global_para)
                global_result, group_result, dp, self.max_dice_global, self.global_model = \
                    evaluateGlobal(self.global_model, self.args, DataLoader, self.max_dice_global)
                print('[global dice]{:0.4f}\t[DP]{:0.4f}'.format(global_result, dp))
                # save model
                mkdir(self.path + '/model/')

                # update client
                for index in range(0, 3):
                    self.model_list[index].load_state_dict(global_para)
                    self.logger_list[index].add_scalar('result/global', global_result, global_step=epoch)
                    self.logger_list[index].add_scalar('result/group', group_result[index], global_step=epoch)
                    self.logger_list[index].add_scalar('result/dp', dp, global_step=epoch)
                self.writer.writerow([str(global_result), str(dp), str(group_result[0]), str(group_result[1]), str(group_result[2])])
                print("*** FairMixup Complete ***")

            # Mixup
            alpha = 1
            gamma = beta(alpha, alpha)

            # init iter
            data_iter1 = iter(self.loader_list[0])
            data_iter2 = iter(self.loader_list[1])
            data_iter3 = iter(self.loader_list[2])
            
            for index in range(0, 3):
                                        
                for i, (image, mask) in enumerate(self.loader_list[index]):
                    try:
                        # get loader data(batch)
                        loss_reg = 0
                        batch_x1, batch_y1 = next(data_iter1)
                        batch_x2, batch_y2 = next(data_iter2)
                        batch_x3, batch_y3 = next(data_iter3)

                        batch_x1, batch_y1 = batch_x1.cuda(), batch_y1.cuda()
                        batch_x2, batch_y2 = batch_x2.cuda(), batch_y2.cuda()
                        batch_x3, batch_y3 = batch_x3.cuda(), batch_y3.cuda()

                        batch_x_mix = [
                            batch_x1 * gamma + batch_x2 * (1 - gamma),
                            batch_x2 * gamma + batch_x3 * (1 - gamma),
                            batch_x3 * gamma + batch_x1 * (1 - gamma)
                        ]
                        
                        # reg on grad
                        batch_x = batch_x_mix[index]
                        # reg loss
                        batch_x = batch_x.requires_grad_(True)
                        output = self.model_list[index](batch_x)
                        gradx = torch.autograd.grad(output.sum(), batch_x, create_graph=True)[0].view(batch_x.shape[0], -1)
                        batch_x_d = batch_x - [batch_x1, batch_x2, batch_x3][index]
                        batch_x_d = batch_x_d.view(batch_x.shape[0], -1)
                        grad_inn = (gradx * batch_x_d).sum(1).view(-1)
                        E_grad = grad_inn.mean()
                        loss_reg += torch.abs(E_grad)

                    except:
                        loss_reg += 0
                    
                    
                    ## get data, pred, calculate loss
                    image, mask = image.cuda(), mask.cuda()
                    pred = self.model_list[index](image)
                    pred = F.interpolate(pred, size=352, mode='bilinear')[:, 0]
                    
                    ## loss_ce + loss_dice
                    loss_ce = F.binary_cross_entropy_with_logits(pred, mask)
                    pred = torch.sigmoid(pred)
                    inter = (pred * mask).sum(dim=(1, 2))
                    union = (pred + mask).sum(dim=(1, 2))
                    loss_dice = 1 - (2 * inter / (union + 1)).mean()
                    loss = loss_ce + loss_dice + loss_reg * self.args.penalty_weight

                    ## backward
                    self.optimizer_list[index].zero_grad()
                    loss.backward()
                    clip_gradient(self.optimizer_list[index], self.args.clip)
                    self.optimizer_list[index].step()
                    global_step_list[index] += 1
                    self.logger_list[index].add_scalar('lr', self.optimizer_list[index].param_groups[0]['lr'],
                                                       global_step=global_step_list[index])
                    self.logger_list[index].add_scalar('loss/loss_ce', loss_ce, global_step=global_step_list[index])
                    self.logger_list[index].add_scalar('loss/loss_dice', loss_dice, global_step=global_step_list[index])

                ## print loss
                print(
                    '[Client{}/3]{} epoch={:03d}/{:03d}, loss_ce={:0.4f}, loss_dice={:0.4f}, test result={:0.4f}'.format(
                        index + 1, datetime.now(), epoch + 1, self.args.epoch, loss_ce.item(), loss_dice.item(),
                        result))
                self.lr_scheduler_list[index].step()
                # save model
                # mkdir(self.path + '/model/')

            com_counter += 1

    def evaluateSub(self, current_model_index):
        dice, iou, cnt = 0, 0, 0
        for index in range(0, 3):
            self.model_list[current_model_index].eval()
            with torch.no_grad():
                self.args.datapath = self.args.datapaths[index]
                data = TestData(self.args)
                loader = DataLoader(dataset=data, batch_size=1, shuffle=False)
                for image, mask, name in loader:
                    image, mask = image.cuda().float(), mask.cuda().float()
                    B, C, H, W = image.shape
                    pred = self.model_list[current_model_index](image)
                    pred = F.interpolate(pred, size=(H, W), mode='bilinear')
                    pred = (pred.squeeze() > 0)
                    inter, union = (pred * mask).sum(dim=(1, 2)), (pred + mask).sum(dim=(1, 2))
                    dice += ((2 * inter + 1) / (union + 1)).sum().cpu().numpy()
                    iou += ((inter + 1) / (union - inter + 1)).sum().cpu().numpy()
                    cnt += B

        if dice / cnt > self.max_dice_list[current_model_index]:
            self.max_dice_list[current_model_index] = dice / cnt
        self.model_list[current_model_index].train()
        return dice / cnt
