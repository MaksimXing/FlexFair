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


class DataKVA(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = [name for name in os.listdir(args.datapath+'Public_5/train/image/site_KVA')]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
        self.color_dict = []
        with open(self.args.datapath+'Public_5/train/train_mean_std.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # belong to KVA
                if int(row[7]) == 1:
                    self.color_dict.append([float(row[1]), float(row[2]), float(row[3]),
                                            float(row[4]), float(row[5]), float(row[6])])


    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.args.datapath+'Public_5/train/image/site_KVA/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if self.args.use_color_exchange:
            mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
            mean_std_2 = random.choice(self.color_dict)
            mean2 = mean_std_2[0:3]
            std2 = mean_std_2[3:6]
            image = np.uint8((image - mean) / std * std2 + mean2)
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

        mask  = cv2.imread(self.args.datapath+'Public_5/train/mask/site_KVA/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        pair  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)
