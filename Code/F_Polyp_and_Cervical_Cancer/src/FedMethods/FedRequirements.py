import os
import sys
import cv2
import datetime
import argparse
import numpy as np
import ctypes
import time

# libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from FedBasicFunc.Data import *
from FedBasicFunc.Models import *
from FedBasicFunc.FedTestKVA_CVC import *
from FedBasicFunc.FedTestKVA_CVC_Mixup import *
from FedBasicFunc.FedTestPrivate import *

from itertools import cycle


def binary_conversion(var: int):
    assert isinstance(var, int)
    if var <= 1024:
        return round(var / 1024, 2)
    else:
        return round(var / (1024 ** 2), 2)
