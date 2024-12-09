import os

import torch

from FedMethods.FedRequirements import *


def get_bce_dice_per_sample(predictions, masks):
    loss_ce, loss_dice = [], []
    for pred, mk in zip(predictions, masks):
        l_ce, l_dice = bce_dice(pred.unsqueeze(0), mk.unsqueeze(0))
        loss_ce.append(l_ce)
        loss_dice.append(l_dice)
    return loss_ce, loss_dice


class FedAvgOOD_Weighted_Public_REx(object):
    def __init__(self, DataList, FedAvgModel, args):
        ## dataset
        self.args = args
        self.DataList = []
        self.DataList.append(DataCVC)
        self.DataList.append(DataKVA)
        self.args.site_num = 2

        self.ret_0_list = []
        self.ret_1_list = []

        self.loader_list = []
        self.max_dice_test = list(0 for i in range(len(self.DataList)))
        # 第一个数据集大小550，第二个900，按这个比例(11:18)分batch_size
        dataset_size = [550, 900]
        self.set_batch_size = []
        for size_ in dataset_size:
            self.set_batch_size.append(round(size_ * 1.0 / args.fairness_step))
        self.total_batch_size = sum(self.set_batch_size)
        for site in range(len(self.DataList)):
            self.loader_list.append(DataLoader(self.DataList[site](args), batch_size=self.set_batch_size[site],
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
        self.local_time = now.strftime("%m%d %H%M%S")

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
                SummaryWriter(args.savepath + self.local_time + '/log/Client' + str(model_index + 1)))

    def train(self):
        global_step = 0

        # Fed counter
        com_counter = 0

        # get dataset size
        dataset_size = []
        dataset_total_size = 0
        for loader in self.loader_list:
            dataset_size.append(len(loader.dataset))
            dataset_total_size += len(loader.dataset)

        for epoch in range(self.args.epoch):

            # Fed
            global_para = self.global_model.state_dict()
            # Init
            if com_counter == 0:
                if self.args.is_same_initial:
                    # replace client with global(for init)
                    for model_index in range(self.args.site_num):
                        self.sub_model_list[model_index].load_state_dict(global_para)
            # for every epoch, every sub model run once
            elif com_counter % self.args.com_round == 0:
                # calculate communicate cost
                sub_net_size = 0
                for model_index in range(self.args.site_num):
                    sub_net_size += sys.getsizeof(self.sub_model_list[model_index].state_dict())
                print("*** FedAvg Execute ***")
                aggregation_start_time = time.process_time()
                for model_index in range(self.args.site_num):
                    sub_net_para = self.sub_model_list[model_index].state_dict()
                    # Calculate Federated
                    if model_index == 0:
                        for key in sub_net_para:
                            global_para[key] = sub_net_para[key] \
                                               * (dataset_size[model_index] * 1.0 / dataset_total_size)
                    else:
                        for key in sub_net_para:
                            global_para[key] += sub_net_para[key] \
                                                * (dataset_size[model_index] * 1.0 / dataset_total_size)
                # update global
                self.global_model.load_state_dict(global_para)
                # update client
                for model_index in range(self.args.site_num):
                    self.sub_model_list[model_index].load_state_dict(global_para)

                aggregation_end_time = time.process_time()

                cur_model_path = self.args.savepath + self.local_time + '/model/'
                if not os.path.exists(cur_model_path):
                    os.makedirs(cur_model_path)
                # torch.save(self.global_model.state_dict(), cur_model_path + 'model_global-epoch=' + str(epoch + 1))
                ret_0, ret_1 = RunFedTestKVA_CVC(self.global_model,
                                                 self.args, cur_model_path, epoch)
                f = open(cur_model_path + "global.csv", 'a+')
                writer = csv.writer(f)
                writer.writerow([str(ret_0[0]), str(ret_0[1]), str(ret_0[2]), str(ret_0[3]), str(ret_0[4])])
                f.close()
                print("*** FedAvg Complete ***")

            dataloader_iterator_list = []
            StopIteration_mark = []
            for loader_index in range(len(self.loader_list)):
                dataloader_iterator_list.append(iter(self.loader_list[loader_index]))
                StopIteration_mark.append(False)


            while True:
                image_list = []
                mask_list = []
                # dataset prepare
                ################################# loader #################################
                for loader_index in range(len(self.loader_list)):
                    try:
                        image_, mask_ = next(dataloader_iterator_list[loader_index])
                    except StopIteration:
                        dataloader_iterator_list[loader_index] = iter(self.loader_list[loader_index])
                        image_, mask_ = next(dataloader_iterator_list[loader_index])
                        StopIteration_mark[loader_index] = True

                    # all iter complete
                    if all(element == True for element in StopIteration_mark):
                        break

                    image_, mask_ = image_.cuda().float(), mask_.cuda().float()
                    rand = int(np.random.choice([256, 288, 320, 352], p=[0.1, 0.2, 0.3, 0.4]))
                    # rand = 256
                    image_ = F.interpolate(image_, size=(rand, rand), mode='bilinear')
                    mask_ = F.interpolate(mask_.unsqueeze(1), size=(rand, rand), mode='nearest').squeeze(1)
                    image_list.append(image_)
                    mask_list.append(mask_)

                # test again
                if all(element == True for element in StopIteration_mark):
                    break

                img_1, img_2 = image_list[0], image_list[1]
                mk_1, mk_2 = mask_list[0], mask_list[1]
                loss_ce_1, loss_dice_1, loss_ce_2, loss_dice_2 = [], [], [], []
                pred_1 = self.sub_model_list[0](img_1)
                pred_2 = self.sub_model_list[1](img_2)
                pred_1 = F.interpolate(pred_1, size=mk_1.shape[1:], mode='bilinear', align_corners=True)[:, 0,
                         :, :]
                lce_1, ldice_1 = get_bce_dice_per_sample(pred_1, mk_1)
                pred_2 = F.interpolate(pred_2, size=mk_2.shape[1:], mode='bilinear', align_corners=True)[
                         :, 0, :, :]
                lce_2, ldice_2 = get_bce_dice_per_sample(pred_2, mk_2)
                loss_ce_1.extend(lce_1)
                loss_ce_2.extend(lce_2)
                loss_dice_1.extend(ldice_1)
                loss_dice_2.extend(ldice_2)
                
                len1 = len(loss_ce_1)
                len2 = len(loss_ce_2)
                total_loss = []
                total_loss.extend(loss_ce_1)
                total_loss.extend(loss_ce_2)
                total_loss = torch.stack(total_loss, dim=0)
                mean = torch.mean(total_loss)   
                
                loss_ce_1 = torch.stack(loss_ce_1, dim=0)
                loss_ce_1 = torch.mean(loss_ce_1)
                
                loss_ce_2 = torch.stack(loss_ce_2, dim=0)
                loss_ce_2 = torch.mean(loss_ce_2)
                
                loss_dice_1 = torch.stack(loss_dice_1, dim=0)
                loss_dice_1 = torch.mean(loss_dice_1)
                loss_dice_2 = torch.stack(loss_dice_2, dim=0)
                loss_dice_2 = torch.mean(loss_dice_2)
                
                weight_list = []
                for weight_index in range(len(self.set_batch_size)):
                    weight_list.append(self.set_batch_size[weight_index] * 1.0 / self.total_batch_size)
                # losses = torch.stack([loss_ce_1 * weight_list[0], loss_ce_2 * weight_list[1]], dim=0)
                losses = torch.stack([loss_ce_1, loss_ce_2], dim=0)
                # mean_orig = torch.mean(losses)
                
                print(StopIteration_mark)
                if len1 == 69 and len2 == 112:
                    penalty = ((losses - mean) ** 2).mean()
                else:
                    penalty = 0
                # total_loss = (loss_ce_1 + loss_dice_1) * weight_list[0] \
                #              + (loss_ce_2 + loss_dice_2) * weight_list[1] \
                #              + penalty * self.args.penalty_weight
                total_loss = (loss_ce_1 + loss_dice_1)\
                             + (loss_ce_2 + loss_dice_2) \
                             + penalty * self.args.penalty_weight
                
                self.logger_list[0].add_scalar('ood/ce_loss', loss_ce_1, global_step)
                self.logger_list[1].add_scalar('ood/ce_loss', loss_ce_2, global_step)
                self.logger_list[0].add_scalar('ood/dice_loss', loss_dice_1, global_step)
                self.logger_list[1].add_scalar('ood/dice_loss', loss_dice_2, global_step)
                self.logger_list[0].add_scalar('ood/penalty', penalty, global_step)
                self.logger_list[1].add_scalar('ood/penalty', penalty, global_step)
                self.optimizer_list[0].zero_grad()
                self.optimizer_list[1].zero_grad()
                total_loss.backward()
                self.optimizer_list[0].step()
                self.optimizer_list[1].step()
                global_step += 1

            print('[Epoch %d/%d]%s' % (
                epoch + 1, self.args.epoch, datetime.datetime.now()))
            for loader_model_index in range(self.args.site_num):
                if epoch + 1 > self.args.start_test_epoch:
                    cur_model_path = self.args.savepath + self.local_time + '/model/Client' + str(
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
                    cur_path = self.args.savepath + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_path):
                        os.makedirs(cur_path)
                    # torch.save(self.sub_model_list[loader_model_index].state_dict(),
                    #            self.args.savepath + '/FedAvg_OOD/' + self.local_time + '/model/Client' + str(
                    #                loader_model_index + 1) + '/model-epoch=' + str(epoch + 1))

            
            com_counter += 1

        log_file = open(self.args.savepath + self.local_time + '/args.txt', "w")
        for arg in vars(self.args):
            log_file.write(str(arg) + ':' + str(getattr(self.args, arg)) + '\n')
        for site in range(self.args.site_num):
            print("[FedAvg+OOD] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
            log_file.write("[FedAvg+OOD] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
            for ret_index in range(len(self.ret_0_list)):
                log_file.write("[ret_0]Epoch:" + str(ret_index +1) + "\t" + str(self.ret_0_list[ret_index]))
                log_file.write("[ret_1]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_1_list[ret_index]))
        log_file.close()
