from FedBasicFunc.FedTestData import *
from FedBasicFunc.Data.FedData import *


class FedTest(object):
    def __init__(self, FedTestData, TrainingModel, args):
        ## dataset
        self.args = args
        self.data = FedTestData(args)
        self.loader = DataLoader(self.data, batch_size=1, pin_memory=True, shuffle=True, num_workers=self.args.num_workers)
        ## model
        self.model = TrainingModel
        self.model.train(False)
        self.model.cuda()

    def prediction(self):
        # print('///testing on [' + self.args.test_dataset + ']///')
        # 定义一个空列表来存储IOU和Dice指标
        iou_list = []
        dice_list = []
        with toßrch.no_grad():
            for image, mask, shape, name, origin, gt in self.loader:
                image = image.cuda().float()
                pred = self.model(image)
                pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
                pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
                pred = torch.sigmoid(pred).cpu().numpy() * 255
                final_pred = np.round(pred)
                if self.args.use_private_dataset:
                    iou_list.append(iou(final_pred, gt))
                    dice_list.append(dice(final_pred, gt))
                if self.args.use_public_dataset:
                    iou_list.append(iou(final_pred, gt))
                    dice_list.append(dice(final_pred, gt))

        mean_iou = np.mean(iou_list)
        mean_dice = np.mean(dice_list)
        # print("平均IOU:", mean_iou)
        # print("平均Dice:", mean_dice)
        self.model.train(True)
        # print('///testing complete///')
        return mean_iou, mean_dice


def RunFedTest(TrainingModel, args, Data):
    if isinstance(Data(args), DataCVC) or isinstance(Data(args), DataCVC_Aeq0) or isinstance(Data(args), DataCVC_Aeq1):
        t = FedTest(FedTestDataCVC, TrainingModel, args)
    elif isinstance(Data(args), DataKVA) or isinstance(Data(args), DataKVA_Aeq0) or isinstance(Data(args), DataKVA_Aeq1):
        t = FedTest(FedTestDataKVA, TrainingModel, args)
    mean_iou, mean_dice = t.prediction()
    return mean_iou, mean_dice
