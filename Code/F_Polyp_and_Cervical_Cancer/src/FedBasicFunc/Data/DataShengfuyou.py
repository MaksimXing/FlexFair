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


class DataShengfuyou(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = [name for name in os.listdir(args.datapath + 'Private/train/shengfuyou/')
                        if 'label' not in name]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
        # self.color_dict = []
        # with open(self.args.datapath + 'Private/train/shengfuyou_train_mean_std.csv', 'r') as file:
        #     reader = csv.reader(file)
        #     for row in reader:
        #         if len(row) != 0:
        #             self.color_dict.append([str(row[0]), float(row[1]), float(row[2])])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.args.datapath + 'Private/train/shengfuyou/' + name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.args.datapath + 'Private/train/shengfuyou/' + name[:-4] + '_label.png',
                          cv2.IMREAD_GRAYSCALE) / 255.0

        # if self.args.use_private_color_exchange:
        #     mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
        #     mean_std_2 = random.choice(self.color_dict)
        #     mean2 = mean_std_2[1]
        #     std2 = mean_std_2[2]
        #     value = (image - mean) / std * std2 + mean2
        #     for row in value:
        #         for i in range(len(row)):
        #             if row[i] < 0:
        #                 row[i] = 0
        #             elif row[i] > 255:
        #                 row[i] = 255
        #     image = np.uint8(value)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)
