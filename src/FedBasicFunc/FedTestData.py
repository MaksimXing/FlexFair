import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from model import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2

from FedBasicFunc.FedIoU import *
from FedBasicFunc.FedTestData import *
from PIL import Image

class FedTestDataCVC(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = [name for name in os.listdir(args.test_data_path + 'Public_5/test/CVC-300/images')]
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.args.test_data_path+'Public_5/test/CVC-300/images/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin = image
        H, W, C = image.shape
        mask = cv2.imread(self.args.test_data_path + 'Public_5/test/CVC-300/masks/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        gt = np.array(Image.open(self.args.test_data_path + 'Public_5/test/CVC-300/masks/' + name))
        gt = gt[:,:,0]
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], (H, W), name, origin, gt

    def __len__(self):
        return len(self.samples)

class FedTestDataKVA(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = [name for name in os.listdir(args.test_data_path + 'Public_5/test/Kvasir/images/')]
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.args.test_data_path+'Public_5/test/Kvasir/images/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin = image
        H, W, C = image.shape
        mask = cv2.imread(self.args.test_data_path + 'Public_5/test/Kvasir/masks/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        gt = np.array(Image.open(self.args.test_data_path + 'Public_5/test/Kvasir/masks/' + name))
        gt = gt[:,:,0]
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], (H, W), name, origin, gt

    def __len__(self):
        return len(self.samples)
