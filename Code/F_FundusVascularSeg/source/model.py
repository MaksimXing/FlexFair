import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Res2Net50, weight_init


class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4 = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear')
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        out = x2 * x3 * x4
        return out

    def initialize(self):
        weight_init(self)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.backbone = Res2Net50()
        channels = [256, 512, 1024, 2048]
        self.fusion = Fusion(channels)
        self.linear = nn.Conv2d(64, 1, kernel_size=1)

        ## initialize
        if args.mode == 'train':
            weight_init(self)
        elif args.mode == 'test':
            self.load_state_dict(torch.load(args.snapshot))
        else:
            raise ValueError

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        pred = self.fusion(x1, x2, x3, x4)
        pred = self.linear(pred)
        return pred
