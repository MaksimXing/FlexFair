import numpy as np
import os

from PIL import Image

# 定义一个函数来计算IOU
def iou(pred, gt):
    # 将预测的mask和GT的mask转换为二值矩阵
    pred = np.array(pred / 255. > 0.5, dtype=np.int_)
    gt = np.array(gt / 255. > 0.5, dtype=np.int_)
    # 计算交集和并集
    intersection = np.sum(pred * gt)
    union = np.sum(pred + gt > 0)
    # 计算IOU
    iou = intersection / (union + 1e-10)  # 防止除零错误

    return iou


# 定义一个函数来计算Dice系数
def dice(pred, gt):
    # 将预测的mask和GT的mask转换为二值矩阵
    pred = np.array(pred / 255. > 0.5, dtype=np.int_)
    gt = np.array(gt / 255. > 0.5, dtype=np.int_)
    # 计算交集和像素和
    intersection = np.sum(pred * gt)
    pixel_sum = np.sum(pred + gt)
    # 计算Dice系数
    dice = 2 * intersection / (pixel_sum + 1e-10)  # 防止除零错误
    return dice


# 假设你有一个prediction文件夹和一个masks文件夹，分别包含预测的mask和GT的mask
# 假设每个文件夹中有10个文件，文件名为"mask_0.png"到"mask_9.png"
# 假设每个文件是一个灰度图像，像素值在0到255之间
# 导入PIL库来读取图像文件
# def FedIou():
#     # 定义一个空列表来存储IOU和Dice指标
#     iou_list = []
#     dice_list = []
#     gt_path = '/home/xinghuijun/MedSeg/SANet-main/data/test.py/CVC-ClinicDB/masks/'
#     pred_path = '/home/xinghuijun/MedSeg/SANet-main/eval/prediction/SANet/CVC-ClinicDB/'
#     labels = os.listdir(gt_path)
#     preds = os.listdir(pred_path)
#     print("len of labels equal to predictions:", len(labels) == len(preds))
#     # 遍历每个文件名
#     for i in range(len(labels)):
#         # 拼接文件路径
#         pred = os.path.join(pred_path, preds[i])
#         gt = os.path.join(gt_path, labels[i])
#         # 使用PIL库读取图像文件，并转换为numpy数组
#         pred = np.array(Image.open(pred))
#         gt = np.array(Image.open(gt))
#         # gt = gt[:,:,0]
#         # 调用iou和dice函数计算指标，并添加到列表中
#         iou_list.append(iou(pred, gt))
#         dice_list.append(dice(pred, gt))
#
#     # 计算平均IOU和平均Dice指标，并打印结果
#     mean_iou = np.mean(iou_list)
#     mean_dice = np.mean(dice_list)
#     print("平均IOU:", mean_iou)
#     print("平均Dice:", mean_dice)

