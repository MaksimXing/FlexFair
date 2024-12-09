import csv
import numpy as np
from FedBasicFunc.FedTestData import *
import cv2


def ThroughDict(dict_pred, xx_dict_A, pred, pred_xx, pred_Aeq0, pred_Aeq0_xx, pred_Aeq1, pred_Aeq1_xx):
    for name_key, res in dict_pred.items():
        try:
            A_value = xx_dict_A[name_key]
        except KeyError:
            continue
        pred.append(res)
        pred_xx.append(res)
        if A_value == 0:
            pred_Aeq0.append(res)
            pred_Aeq0_xx.append(res)
        elif A_value == 1:
            pred_Aeq1.append(res)
            pred_Aeq1_xx.append(res)
    return pred, pred_xx, pred_Aeq0, pred_Aeq0_xx, pred_Aeq1, pred_Aeq1_xx


class FedTestMixed(object):
    def __init__(self, FedTestData, TrainingModel, writer, args):
        ## dataset
        self.args = args
        self.data = FedTestData(args)
        self.loader = DataLoader(self.data, batch_size=1, pin_memory=True, shuffle=True, num_workers=0)
        ## model
        self.model = TrainingModel
        self.model.train(False)
        self.model.cuda()
        self.writer = writer

    def prediction(self):
        # print('///testing on [' + self.args.test_dataset + ']///')
        # Save IoU and Dice
        iou_list = []
        dice_list = []
        with torch.no_grad():
            dict_out = {}
            for image, mask, shape, name, origin, gt in self.loader:
                image = image.cuda().float()
                pred = self.model(image)
                pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
                pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
                pred = torch.sigmoid(pred).cpu().numpy() * 255
                final_pred = np.round(pred)
                iou_ = iou(final_pred, gt)
                dice_ = dice(final_pred, gt)
                # cv2.imwrite('C:/Users/xingh/Desktop/' + name[0], final_pred)
                if self.writer is not None:
                    self.writer.writerow([str(name[0][:-4]), str(iou_), str(dice_)])
                dict_out[str(name[0][:-4])] = dice_
                iou_list.append(iou_)
                dice_list.append(dice_)

        mean_iou = np.mean(iou_list)
        mean_dice = np.mean(dice_list)
        len_dice = len(dice_list)
        self.model.train(True)
        return mean_iou, mean_dice, len_dice



def RunNewFedTestMixed(args, TrainingModel, cur_model_path, cur_epoch):
    f = open(cur_model_path + "epoch_" + str(cur_epoch + 1) + "_PREC.csv", 'w')
    writer = csv.writer(f)
    t1 = FedTestMixed(FedTestDataNewZhongyi, TrainingModel, writer, args)
    t2 = FedTestMixed(FedTestDataNewZhongda, TrainingModel, writer, args)
    t4 = FedTestMixed(FedTestDataNewZhongzhong, TrainingModel, writer, args)
    t3 = FedTestMixed(FedTestDataShengfuyou, TrainingModel, writer, args)

    
    _, mean_dice1, len_dice1 = t1.prediction()
    _, mean_dice2, len_dice2 = t2.prediction()
    _, mean_dice3, len_dice3 = t3.prediction()
    _, mean_dice4, len_dice4 = t4.prediction()
    total_len = len_dice1 + len_dice2 + len_dice3 + len_dice4
    global_dice = (mean_dice1 * len_dice1 + mean_dice2 * len_dice2 + mean_dice3 * len_dice3 + mean_dice4 * len_dice4) / total_len
    mean_dice = [mean_dice1, mean_dice2, mean_dice3, mean_dice4]
    
    f.close()
    return mean_dice, global_dice

def RunFedTestPrivateBaseline(TrainingModel, args, cur_model_path, cur_epoch):
    mean_dice, global_dice = RunNewFedTestMixed(args, TrainingModel, cur_model_path, cur_epoch)

    return mean_dice, global_dice



def RunNewFedTestMixed_client(args, TrainingModel, cur_model_path, cur_epoch):
    writer = None
    t1 = FedTestMixed(FedTestDataNewZhongyi, TrainingModel, writer, args)
    t2 = FedTestMixed(FedTestDataNewZhongda, TrainingModel, writer, args)
    t4 = FedTestMixed(FedTestDataNewZhongzhong, TrainingModel, writer, args)
    t3 = FedTestMixed(FedTestDataShengfuyou, TrainingModel, writer, args)

    
    _, mean_dice1, len_dice1 = t1.prediction()
    _, mean_dice2, len_dice2 = t2.prediction()
    _, mean_dice3, len_dice3 = t3.prediction()
    _, mean_dice4, len_dice4 = t4.prediction()
    total_len = len_dice1 + len_dice2 + len_dice3 + len_dice4
    global_dice = (mean_dice1 * len_dice1 + mean_dice2 * len_dice2 + mean_dice3 * len_dice3 + mean_dice4 * len_dice4) / total_len
    mean_dice = [mean_dice1, mean_dice2, mean_dice3, mean_dice4]
    
    return mean_dice, global_dice

def RunFedTestPrivateBaseline_client(TrainingModel, args, cur_model_path, cur_epoch):
    mean_dice, global_dice = RunNewFedTestMixed_client(args, TrainingModel, cur_model_path, cur_epoch)

    return mean_dice, global_dice 