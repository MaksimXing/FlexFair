from FedBasicFunc.Data.FedData import *
from FedBasicFunc.FedTestData import *

sites = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']


class FedTestDataKVA_CVC(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = [name for name in os.listdir(args.test_data_path + 'Public_5/test/CVC_KVA/images')]
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.args.test_data_path+'Public_5/test/CVC_KVA/images/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin = image
        H, W, C = image.shape
        mask = cv2.imread(self.args.test_data_path + 'Public_5/test/CVC_KVA/masks/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        gt = np.array(Image.open(self.args.test_data_path + 'Public_5/test/CVC_KVA/masks/' + name))
        gt = gt[:,:,0]
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], (H, W), name, origin, gt

    def __len__(self):
        return len(self.samples)


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
        # 定义一个空列表来存储IOU和Dice指标
        iou_list = []
        dice_list = []
        with torch.no_grad():
            dict_out = {}
            for image, mask, shape, name, origin, gt in self.loader:
                image = image.cuda().float()
                pred, pred_ = self.model(image)
                pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
                pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
                pred = torch.sigmoid(pred).cpu().numpy() * 255
                final_pred = np.round(pred)
                iou_ = iou(final_pred, gt)
                dice_ = dice(final_pred, gt)
                self.writer.writerow([str(name[0][:-4]), str(iou_), str(dice_)])
                dict_out[str(name[0][:-4])] = dice_
                iou_list.append(iou_)
                dice_list.append(dice_)

        mean_iou = np.mean(iou_list)
        mean_dice = np.mean(dice_list)
        self.model.train(True)
        return mean_iou, mean_dice


def RunFedTestMixed(args, TrainingModel, cur_model_path, cur_epoch):
    f = open(cur_model_path + "epoch_" + str(cur_epoch + 1) + "_PREC.csv", 'w')
    writer = csv.writer(f)
    t1 = FedTestMixed(FedTestDataKVA_CVC, TrainingModel, writer, args)
    mean_iou, mean_dice = t1.prediction()
    f.close()
    return mean_iou, mean_dice


def calculateDP_PUBLIC(args, cur_model_path, cur_epoch):
    dict_A = {}
    dict_A_CVC300 = {}
    dict_A_Kvasir = {}
    dict_pred = {}

    with open(args.datapath + 'Public_5/test/test_mean.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 0:
                if row[0] == sites[0]:
                    dict_A_CVC300[row[1][:-4]] = int(row[2])
                elif row[0] == sites[4]:
                    dict_A_Kvasir[row[1][:-4]] = int(row[2])

    with open(cur_model_path + "epoch_" + str(cur_epoch + 1) + "_PREC.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 0:
                dict_pred[row[0]] = float(row[2])

    # DP
    pred_Aeq0 = []
    pred_Aeq1 = []
    pred_KVA = []
    pred_CVC = []
    pred = []
    pred_Aeq0_KVA = []
    pred_Aeq1_KVA = []
    pred_Aeq0_CVC = []
    pred_Aeq1_CVC = []

    for name_key, res in dict_pred.items():
        try:
            A_value = dict_A_Kvasir[name_key]
        except KeyError:
            continue
        pred_KVA.append(res)
        pred.append(res)
        if A_value == 0:
            pred_Aeq0.append(res)
            pred_Aeq0_KVA.append(res)
        elif A_value == 1:
            pred_Aeq1.append(res)
            pred_Aeq1_KVA.append(res)


    for name_key, res in dict_pred.items():
        try:
            A_value = dict_A_CVC300[name_key]
        except KeyError:
            continue
        pred_CVC.append(res)
        pred.append(res)
        if A_value == 0:
            pred_Aeq0.append(res)
            pred_Aeq0_CVC.append(res)
        elif A_value == 1:
            pred_Aeq1.append(res)
            pred_Aeq1_CVC.append(res)


    return [np.mean(pred), np.mean(pred_CVC), np.mean(pred_KVA), np.mean(pred_Aeq0), np.mean(pred_Aeq1)],\
           [np.mean(pred_Aeq0_CVC), np.mean(pred_Aeq1_CVC), np.mean(pred_Aeq0_KVA), np.mean(pred_Aeq1_KVA)]


def RunFedTestKVA_CVC_Mixup(TrainingModel, args, cur_model_path, cur_epoch):
    RunFedTestMixed(args, TrainingModel, cur_model_path, cur_epoch)
    ret_0, ret_1 = calculateDP_PUBLIC(args, cur_model_path, cur_epoch)
    return ret_0, ret_1