import os

from FedMethods.FedRequirements import *
from FedBasicFunc.Models.FedAvgModel import FedAvgModel


class FedAvgTrainPublic(object):
    def __init__(self, DataList, FedAvgModel, args):
        ## dataset
        self.args = args
        self.DataList = DataList

        self.ret_0_list = []
        self.ret_1_list = []

        self.loader_list = []
        self.max_dice_test = list(0 for i in range(args.site_num))
        self.max_iou_test = list(0 for i in range(args.site_num))
        self.max_dice_val = list(0 for i in range(args.site_num))
        self.max_iou_val = list(0 for i in range(args.site_num))
        for site in range(args.site_num):
            self.loader_list.append(DataLoader(DataList[site](args), batch_size=args.batch_size,
                                               pin_memory=True, shuffle=True,
                                               num_workers=args.num_workers))

        ## GLOBAL model
        self.global_model = FedAvgModel(args)
        self.global_model.train(True)
        self.global_model.cuda()

        ## SUB model
        self.sub_model_list = []
        for site in range(args.site_num):
            self.sub_model_list.append(FedAvgModel(args))
            self.sub_model_list[site].train(True)
            self.sub_model_list[site].cuda()

        now = datetime.datetime.now()
        self.local_time = now.strftime("%Y-%m-%d %H_%M_%S")

        ## parameter
        self.optimizer_list = []
        self.logger_list = []
        for model_index in range(args.site_num):
            base, head = [], []
            for name, param in self.sub_model_list[model_index].named_parameters():
                if 'bkbone' in name:
                    base.append(param)
                else:
                    head.append(param)
            self.optimizer_list.append(torch.optim.SGD([{'params': base, 'lr': 0.1 * args.lr},
                                                        {'params': head, 'lr': args.lr}],
                                                       momentum=args.momentum, weight_decay=args.weight_decay,
                                                       nesterov=args.nesterov))
            self.logger_list.append(
                SummaryWriter(args.savepath + '/FedAvg/' + self.local_time + '/log/Client' + str(model_index + 1)))

    def train(self):
        global_step_list = list(0 for i in range(self.args.site_num))

        # Fed counter
        com_counter = 0

        # 求每个数据集长度
        dataset_size = []
        dataset_total_size = 0
        for loader in self.loader_list:
            dataset_size.append(len(loader.dataset))
            dataset_total_size += len(loader.dataset)

        for epoch in range(self.args.epoch):
            if epoch + 1 in [64, 96]:
                self.optimizer_list[loader_model_index].param_groups[0]['lr'] *= 0.5
                self.optimizer_list[loader_model_index].param_groups[1]['lr'] *= 0.5

            # Fed
            global_para = self.global_model.state_dict()
            # Init
            if com_counter == 0:
                if self.args.is_same_initial:
                    # 对子网权重赋和全局权重相同的值
                    for model_index in range(self.args.site_num):
                        self.sub_model_list[model_index].load_state_dict(global_para)
            # for every epoch, every sub model run once
            elif com_counter % self.args.com_round == 0:
                # 计算通信开销
                sub_net_size = 0
                for model_index in range(self.args.site_num):
                    sub_net_size += sys.getsizeof(self.sub_model_list[model_index].state_dict())
                print("*** FedAvg Execute ***")
                aggregation_start_time = time.process_time()
                for model_index in range(self.args.site_num):
                    sub_net_para = self.sub_model_list[model_index].state_dict()
                    # 累加求和，**权重是数据占比**
                    if model_index == 0:
                        for key in sub_net_para:
                            global_para[key] = sub_net_para[key] \
                                               * (dataset_size[model_index] * 1.0 / dataset_total_size)
                    else:
                        for key in sub_net_para:
                            global_para[key] += sub_net_para[key] \
                                                * (dataset_size[model_index] * 1.0 / dataset_total_size)
                # 更新全局模型
                self.global_model.load_state_dict(global_para)
                # 更新子网模型
                for model_index in range(self.args.site_num):
                    self.sub_model_list[model_index].load_state_dict(global_para)

                aggregation_end_time = time.process_time()

                # 计算通信开销
                global_net_size = self.args.site_num * sys.getsizeof(self.global_model.state_dict())
                for loader_index in range(self.args.site_num):
                    self.logger_list[loader_index].add_scalar('time/aggregation_time)',
                                                              aggregation_end_time - aggregation_start_time,
                                                              epoch)
                    self.logger_list[loader_index].add_scalar('communication/sub_model_total_size)',
                                                              sub_net_size,
                                                              epoch)
                    self.logger_list[loader_index].add_scalar('communication/global_model_total_size)',
                                                              global_net_size,
                                                              epoch)
                print("*** FedAvg Complete ***")

            for loader_model_index in range(self.args.site_num):
                epoch_start_time = time.process_time()

                loss_ce = 0
                loss_dice = 0
                for image, mask in self.loader_list[loader_model_index]:
                    single_image_start_time = time.process_time()
                    image, mask = image.cuda().float(), mask.cuda().float()
                    rand = int(np.random.choice([256, 288, 320, 352], p=[0.1, 0.2, 0.3, 0.4]))
                    image = F.interpolate(image, size=(rand, rand), mode='bilinear')
                    mask = F.interpolate(mask.unsqueeze(1), size=(rand, rand), mode='nearest').squeeze(1)

                    pred = self.sub_model_list[loader_model_index](image)
                    pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)[:, 0, :, :]
                    loss_ce, loss_dice = bce_dice(pred, mask)

                    self.optimizer_list[loader_model_index].zero_grad()
                    scale_loss = loss_ce + loss_dice
                    scale_loss.backward()
                    self.optimizer_list[loader_model_index].step()
                    single_image_end_time = time.process_time()

                    ## log
                    global_step_list[loader_model_index] += 1
                    self.logger_list[loader_model_index].add_scalar('lr/lr',
                                                                    self.optimizer_list[
                                                                        loader_model_index].param_groups[0]['lr'],
                                                                    global_step=global_step_list[loader_model_index])
                    self.logger_list[loader_model_index].add_scalar('loss/loss_ce',
                                                                    loss_ce,
                                                                    global_step=global_step_list[loader_model_index])
                    self.logger_list[loader_model_index].add_scalar('loss/loss_dice',
                                                                    loss_dice,
                                                                    global_step=global_step_list[loader_model_index])
                    self.logger_list[loader_model_index].add_scalar('loss/loss_total',
                                                                    scale_loss,
                                                                    global_step=global_step_list[loader_model_index])
                    self.logger_list[loader_model_index].add_scalar('time/single_image_process',
                                                                    single_image_end_time - single_image_start_time,
                                                                    global_step=global_step_list[loader_model_index])

                print('[Client 0%d, Epoch %d/%d]%s | lr=%.6f | ce=%.6f | dice=%.6f' % (
                    loader_model_index + 1, epoch + 1, self.args.epoch, datetime.datetime.now(),
                    self.optimizer_list[loader_model_index].param_groups[0]['lr'], loss_ce.item(), loss_dice.item()))

                if epoch + 1 > self.args.start_test_epoch:
                    cur_model_path = self.args.savepath + '/FedAvg/' + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_model_path):
                        os.makedirs(cur_model_path)
                    ret_0, ret_1 = RunFedTestKVA_CVC(self.sub_model_list[loader_model_index],
                                                     self.args, cur_model_path, epoch)
                    mean_dice_test = ret_0[0]

                    self.logger_list[loader_model_index].add_scalar('test/mean dice', mean_dice_test, epoch + 1)
                    self.logger_list[loader_model_index].add_scalar('test/CVC dice', ret_0[1], epoch + 1)
                    self.logger_list[loader_model_index].add_scalar('test/KVA dice', ret_0[2], epoch + 1)
                    self.logger_list[loader_model_index].add_scalar('test/A=0 dice', ret_0[3], epoch + 1)
                    self.logger_list[loader_model_index].add_scalar('test/A=1 dice', ret_0[4], epoch + 1)

                    self.ret_0_list.append(ret_0)
                    self.ret_1_list.append(ret_1)

                    if mean_dice_test > self.max_dice_test[loader_model_index]:
                        self.max_dice_test[loader_model_index] = mean_dice_test

                if epoch + 1 > self.args.start_save_model and (epoch + 1) % self.args.save_model_per_epoch == 0:
                    cur_path = self.args.savepath + '/FedAvg/' + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_path):
                        os.makedirs(cur_path)
                    torch.save(self.sub_model_list[loader_model_index].state_dict(),
                               self.args.savepath + '/FedAvg/' + self.local_time + '/model/Client' + str(
                                   loader_model_index + 1) + '/model-epoch=' + str(epoch + 1))

                epoch_end_time = time.process_time()
                self.logger_list[loader_model_index].add_scalar('time/epoch_process',
                                                                epoch_end_time - epoch_start_time,
                                                                epoch + 1)

            # 通信轮次计数+1
            com_counter += 1

        log_file = open(self.args.savepath + '/FedAvg/' + self.local_time + '/args.txt', "w")
        for arg in vars(self.args):
            log_file.write(str(arg) + ':' + str(getattr(self.args, arg)) + '\n')
        for site in range(self.args.site_num):
            print("[FedAvg] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
            log_file.write("[FedAvg] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
        for ret_index in range(len(self.ret_0_list)):
            log_file.write("[ret_0]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_0_list[ret_index]) + "\n")
            log_file.write("[ret_1]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_1_list[ret_index]) + "\n")
        log_file.close()
