import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
import csv
import random


class DataNewZhongda(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = [name for name in os.listdir(args.datapath + 'Private/train/zhongda/')
                        if 'label' not in name]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.args.datapath + 'Private/train/zhongda/' + name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.args.datapath + 'Private/train/zhongda/' + name[:-4] + '_label.png',
                          cv2.IMREAD_GRAYSCALE) / 255.0


        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)
