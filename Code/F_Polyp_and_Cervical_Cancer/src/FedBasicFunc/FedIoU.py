import numpy as np
import os

from PIL import Image

def iou(pred, gt):
    pred = np.array(pred / 255. > 0.5, dtype=np.int_)
    gt = np.array(gt / 255. > 0.5, dtype=np.int_)
    intersection = np.sum(pred * gt)
    union = np.sum(pred + gt > 0)
    iou = intersection / (union + 1e-10)

    return iou


def dice(pred, gt):
    pred = np.array(pred / 255. > 0.5, dtype=np.int_)
    gt = np.array(gt / 255. > 0.5, dtype=np.int_)
    intersection = np.sum(pred * gt)
    pixel_sum = np.sum(pred + gt)
    dice = 2 * intersection / (pixel_sum + 1e-10)
    return dice
