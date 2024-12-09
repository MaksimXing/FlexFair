import os
import cv2
from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrainData(Dataset):
    def __init__(self, args):
        self.samples = []
        with open(args.datapath + '/train.txt', 'r') as lines:
            for line in lines:
                image, mask, _ = line.strip().split(' ')
                self.samples.append((args.datapath + '/' + image, args.datapath + '/' + mask))

        self.transform = A.Compose([
            A.Normalize(),
            A.RandomCrop(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image, mask = np.array(Image.open(image_name)), np.array(Image.open(mask_name))
        mask = np.float32(mask > (mask.max() / 2))
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)


class TestData(Dataset):
    def __init__(self, args):
        self.samples = []
        self.args = args
        with open(args.datapath + '/test.txt', 'r') as lines:
            for line in lines:
                image, mask, _ = line.strip().split(' ')
                self.samples.append((args.datapath + '/' + image, args.datapath + '/' + mask))
        # print('Test Data: %s,   Test Samples: %s' % (args.datapath, len(self.samples)))

        self.transform = A.Compose([
            A.Normalize(),
            # A.Resize(352, 352),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image, mask = np.array(Image.open(image_name)), np.array(Image.open(mask_name))
        mask = np.float32(mask > (mask.max() / 2))
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], mask_name

    def __len__(self):
        return len(self.samples)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def evaluateGlobal(global_model, args, DataLoader, max_dice_global):
    dice, iou, cnt = 0, 0, 0
    dice_ = [0, 0, 0]
    cnt_ = [0, 0, 0]
    for index in range(0, 3):
        global_model.eval()
        with torch.no_grad():
            args.datapath = args.datapaths[index]
            data = TestData(args)
            loader = DataLoader(dataset=data, batch_size=1, shuffle=False)
            for image, mask, name in loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                B, C, H, W = image.shape
                pred = global_model(image)
                pred = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred = (pred.squeeze() > 0)
                inter, union = (pred * mask).sum(dim=(1, 2)), (pred + mask).sum(dim=(1, 2))
                res = ((2 * inter + 1) / (union + 1)).sum().cpu().numpy()
                dice += res
                dice_[index] += res
                iou += ((inter + 1) / (union - inter + 1)).sum().cpu().numpy()
                cnt += B
                cnt_[index] += B

    if dice / cnt > max_dice_global:
        max_dice_global = dice / cnt
    global_model.train()
    dice_ = np.array(dice_)
    cnt_ = np.array(cnt_)
    dp = np.std(dice_ / cnt_)
    return dice / cnt, dice_ / cnt_, dp, max_dice_global, global_model