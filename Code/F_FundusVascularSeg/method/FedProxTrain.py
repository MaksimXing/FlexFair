import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

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

plt.ion()

import csv


class FedProxTrain:
    def __init__(self, args):
        ## parameter
        self.args = args

        # logger
        now = datetime.now()
        local_time = now.strftime("%m%d-%H%M%S")
        self.path = args.output_path + local_time
        mkdir(self.path)

        self.logger_list = []
        for index in range(0, 3):
            self.logger_list.append(SummaryWriter(self.path + '/log/' + args.dataset[index]))
        f = open(self.path + '/result.csv', "w", newline="")
        self.writer = csv.writer(f)

        ## model
        self.model_list = []
        for index in range(0, 3):
            self.model_list.append(Model(args).cuda())
            self.model_list[index].train()

        self.global_model = Model(args).cuda()
        self.global_model.train()

        ## data
        self.dataset_list = []
        self.loader_list = []
        for index in range(0, 3):
            args.datapath = args.datapaths[index]
            self.dataset_list.append(TrainData(args))
            self.loader_list.append(DataLoader(dataset=self.dataset_list[index], batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers))
        args.datapath = ''


        ## optimizer
        self.max_dice_list = [0, 0, 0]
        self.max_dice_global = 0
        self.optimizer_list = []
        self.lr_scheduler_list = []
        for index in range(0, 3):
            self.optimizer_list.append(torch.optim.AdamW(self.model_list[index].parameters(), lr=args.lr))
            self.lr_scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_list[index], T_max=args.epoch, eta_min=0))

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

            # Fed
            global_para = self.global_model.state_dict()
            # Init
            if com_counter == 0:
                # replace client with global(for init)
                for index in range(0, 3):
                    self.model_list[index].load_state_dict(global_para)
            elif com_counter % self.args.com_round == 0:
                print("*** FedProx Execute ***")
                for index in range(0, 3):
                    sub_net_para = self.model_list[index].state_dict()
                    # Calculate Federated
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
                mkdir(self.path + '/model/')

                # update client
                for index in range(0, 3):
                    self.model_list[index].load_state_dict(global_para)
                    self.logger_list[index].add_scalar('result/global',global_result,global_step=epoch)
                    self.logger_list[index].add_scalar('result/global',global_result,global_step=epoch)
                    self.logger_list[index].add_scalar('result/global', global_result, global_step=epoch)
                    self.logger_list[index].add_scalar('result/group', group_result[index], global_step=epoch)
                    self.logger_list[index].add_scalar('result/dp', dp, global_step=epoch)
                self.writer.writerow([str(global_result), str(dp), str(group_result[0]), str(group_result[1]), str(group_result[2])])
                print("*** FedProx Complete ***")

            for index in range(0, 3):
                for i, (image, mask) in enumerate(self.loader_list[index]):
                    ## pred
                    image, mask = image.cuda(), mask.cuda()
                    pred = self.model_list[index](image)
                    pred = F.interpolate(pred, size=352, mode='bilinear')[:, 0]
                    ## loss_ce + loss_dice
                    loss_ce = F.binary_cross_entropy_with_logits(pred, mask)
                    pred = torch.sigmoid(pred)
                    inter = (pred * mask).sum(dim=(1, 2))
                    union = (pred + mask).sum(dim=(1, 2))
                    loss_dice = 1 - (2 * inter / (union + 1)).mean()
                    # calculate reg
                    fedprox_reg = 0.0
                    global_weight_collector = list(self.global_model.parameters())
                    for param_index, param in enumerate(self.model_list[index].parameters()):
                        fedprox_reg += ((self.args.miu / 2)
                                        * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss = loss_ce + loss_dice + fedprox_reg

                    ## backward
                    self.optimizer_list[index].zero_grad()
                    loss.backward()
                    clip_gradient(self.optimizer_list[index], self.args.clip)
                    self.optimizer_list[index].step()
                    global_step_list[index] += 1
                    self.logger_list[index].add_scalar('lr', self.optimizer_list[index].param_groups[0]['lr'],
                                                       global_step=global_step_list[index])
                    self.logger_list[index].add_scalar('loss/fedprox_reg',fedprox_reg,global_step=global_step_list[index])
                    self.logger_list[index].add_scalar('loss/loss_ce',loss_ce,global_step=global_step_list[index])
                    self.logger_list[index].add_scalar('loss/loss_dice',loss_dice,global_step=global_step_list[index])

                ## print loss
                print('[Client{}/3]{} epoch={:03d}/{:03d}, loss_ce={:0.4f}, loss_dice={:0.4f}, test result={:0.4f}'.format(
                    index + 1,datetime.now(), epoch + 1, self.args.epoch, loss_ce.item(), loss_dice.item(), result))
                self.lr_scheduler_list[index].step()
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
