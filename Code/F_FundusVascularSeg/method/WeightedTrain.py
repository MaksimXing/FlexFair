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
import csv

plt.ion()

def get_bce_dice_per_sample(predictions, masks):
    loss_ce, loss_dice = [], []
    for pred, mk in zip(predictions, masks):
        l_ce = F.binary_cross_entropy_with_logits(pred, mk)
        pred = torch.sigmoid(pred)
        inter = (pred * mk).sum(dim=(0, 1))
        union = (pred + mk).sum(dim=(0, 1))
        l_dice = 1 - (2 * inter / (union + 1)).mean()
        loss_ce.append(l_ce)
        loss_dice.append(l_dice)
    return loss_ce, loss_dice


class FedAvgOODTrain:
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
        # resize batch
        dataset_size = [21, 20, 15]
        self.set_batch_size = []
        for size_ in dataset_size:
            self.set_batch_size.append(round(size_ * 1.0 / args.fairness_step))
        self.total_batch_size = sum(self.set_batch_size)
        self.dataset_list = []
        self.loader_list = []
        for index in range(0, 3):
            args.datapath = args.datapaths[index]
            self.dataset_list.append(TrainData(args))
            self.loader_list.append(DataLoader(dataset=self.dataset_list[index], batch_size=self.set_batch_size[index],
                                               shuffle=True, num_workers=args.num_workers))
        args.datapath = ''

        ## optimizer
        self.max_dice_list = [0, 0, 0]
        self.max_dice_global = 0
        self.optimizer_list = []
        self.lr_scheduler_list = []
        for index in range(0, 3):
            self.optimizer_list.append(torch.optim.AdamW(self.model_list[index].parameters(), lr=args.lr))
            self.lr_scheduler_list.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_list[index], T_max=args.epoch, eta_min=0))

    def forward(self):
        global_step = 0
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
                print("*** FedAvgOOD Execute ***")
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
                    self.logger_list[index].add_scalar('result/global', global_result, global_step=epoch)
                    self.logger_list[index].add_scalar('result/group', group_result[index], global_step=epoch)
                    self.logger_list[index].add_scalar('result/dp', dp, global_step=epoch)
                self.writer.writerow(
                    [str(global_result), str(dp), str(group_result[0]), str(group_result[1]), str(group_result[2])])
                print("*** FedAvgOOD Complete ***")

            dataloader_iterator_list = []
            StopIteration_mark = []
            for loader_index in range(len(self.loader_list)):
                dataloader_iterator_list.append(iter(self.loader_list[loader_index]))
                StopIteration_mark.append(False)

            index_count = [0, 0, 0]

            while True:
                image_list = []
                mask_list = []
                for loader_index in range(len(self.loader_list)):
                    try:
                        image_, mask_ = next(dataloader_iterator_list[loader_index])
                        index_count[loader_index] += 1
                    except StopIteration:
                        dataloader_iterator_list[loader_index] = iter(self.loader_list[loader_index])
                        image_, mask_ = next(dataloader_iterator_list[loader_index])
                        StopIteration_mark[loader_index] = True
                        index_count[loader_index] = 0

                    # all iter complete
                    if all(element == True for element in StopIteration_mark):
                        break

                    image_, mask_ = image_.cuda(), mask_.cuda()
                    image_list.append(image_)
                    mask_list.append(mask_)

                # test again
                if all(element == True for element in StopIteration_mark):
                    break

                loss_ce = [[], [], []]
                loss_dice = [[], [], []]
                images = [image_list[0], image_list[1], image_list[2]]
                masks = [mask_list[0], mask_list[1], mask_list[2]]

                for loader_model_index in range(3):
                    model = self.model_list[loader_model_index]
                    img_1 = images[loader_model_index]

                    pred_1 = model(img_1)
                    pred_1 = F.interpolate(pred_1, size=352, mode='bilinear')[:, 0]

                    mk_1 = masks[loader_model_index]

                    lce_1, ldice_1 = get_bce_dice_per_sample(pred_1, mk_1)
                    loss_ce[loader_model_index].extend(lce_1)
                    loss_dice[loader_model_index].extend(ldice_1)

                total_loss, len_loss = [], []
                for index in range(3):
                    total_loss.extend(loss_ce[index])
                    len_loss.append(len(loss_ce[index]))
                    
                total_loss = torch.stack(total_loss, dim=0)
                mean = torch.mean(total_loss)   
                
                for index in range(3):
                    loss_ce[index] = torch.stack(loss_ce[index], dim=0)
                    loss_ce[index] = torch.mean(loss_ce[index])
                    loss_dice[index] = torch.stack(loss_dice[index], dim=0)
                    loss_dice[index] = torch.mean(loss_dice[index])

                weight_list = []
                for weight_index in range(len(self.set_batch_size)):
                    weight_list.append(self.set_batch_size[weight_index] * 1.0 / self.total_batch_size)


                losses = torch.stack([loss_ce[0], loss_ce[1], loss_ce[2]], dim=0)

                if len_loss[0] == 4 and len_loss[1] == 4 and len_loss[2] == 3:
                    penalty = ((losses - mean) ** 2).mean()
                else:
                    penalty = 0
                
                loss = (loss_ce[0] + loss_dice[0]) \
                       + (loss_ce[1] + loss_dice[1]) \
                       + (loss_ce[2] + loss_dice[2]) \
                       + penalty * self.args.penalty_weight

                ## backward
                for index in range(3):
                    self.optimizer_list[index].zero_grad()
                loss.backward()
                for index in range(3):
                    clip_gradient(self.optimizer_list[index], self.args.clip)
                    self.optimizer_list[index].step()
                    self.lr_scheduler_list[index].step()
                global_step += 1
                for index in range(3):
                    self.logger_list[index].add_scalar('lr', self.optimizer_list[index].param_groups[0]['lr'],
                                                       global_step=global_step)
                    self.logger_list[index].add_scalar('loss/loss_ce', loss_ce[index], global_step=global_step)
                    self.logger_list[index].add_scalar('loss/loss_dice', loss_dice[index], global_step=global_step)
                    self.logger_list[index].add_scalar('ood/penalty', penalty, global_step=global_step)
            # result = self.evaluateSub(index)
            ## print loss
            print('[OOD]{} epoch={:03d}/{:03d}'.format(datetime.now(), epoch + 1, self.args.epoch))
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
