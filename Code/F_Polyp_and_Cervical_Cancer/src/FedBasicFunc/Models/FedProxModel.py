#coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import Res2Net50, weight_init
from pvtv2 import *


class FedProxModel(nn.Module):
    def __init__(self, args):
        super(FedProxModel, self).__init__()
        self.args    = args
        channels = []
        if args.backbone == 'res2net50':
            self.bkbone = Res2Net50()
            channels = [256, 512, 1024, 2048]
        if args.backbone == 'pvt_v2_b2':
            self.bkbone = pvt_v2_b2()
            channels = [64, 128, 320, 512]
        self.linear5 = nn.Sequential(nn.Conv2d(channels[-1], 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[-2], 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[-3], 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
        pred = self.predict(pred)
        return pred

    def initialize(self):
        if self.args.snapshot:
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)

