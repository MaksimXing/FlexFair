import os

import torch

from FedMethods.FedRequirements import *
from FedBasicFunc.Models.FedAvgModel import FedAvgModel


def get_bce_dice_per_sample(predictions, masks):
    loss_ce, loss_dice = [], []
    for pred, mk in zip(predictions, masks):
        l_ce, l_dice = bce_dice(pred.unsqueeze(0), mk.unsqueeze(0))
        loss_ce.append(l_ce)
        loss_dice.append(l_dice)
    return loss_ce, loss_dice


class FedAvgOOD_Weighted_Private_REx(object):
    def __init__(self, DataList, FedAvgModel, args):
        ## dataset
        self.args = args
        self.DataList = []
        self.DataList.append(DataNewZhongda)
        self.DataList.append(DataShengfuyou)
        self.DataList.append(DataNewZhongzhong)
        self.DataList.append(DataNewZhongyi)
        args.site_num = 4

        self.ret_0_list = []
        self.ret_1_list = []

        self.loader_list = []
        self.max_dice_test = list(0 for i in range(len(self.DataList)))
        # 按这个比例分batch_size
        dataset_size = [1041, 197, 940, 242]
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

                cur_model_path = self.args.savepath + self.local_time + '/model/'
                if not os.path.exists(cur_model_path):
                    os.makedirs(cur_model_path)
                torch.save(self.global_model.state_dict(), cur_model_path + 'model_global-epoch=' + str(epoch + 1))
                RunFedTestPrivateBaseline(self.global_model,
                                                 self.args, cur_model_path, epoch)
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

                    rand = int(np.random.choice([256, 288, 320], p=[0.2, 0.3, 0.5]))
                    image_ = F.interpolate(image_, size=(rand, rand), mode='bilinear')
                    mask_ = F.interpolate(mask_.unsqueeze(1), size=(rand, rand), mode='nearest').squeeze(1)
                    image_list.append(image_)
                    mask_list.append(mask_)

                # test again
                if all(element == True for element in StopIteration_mark):
                    break

                loss_ce = [[], [], [], []]
                loss_dice = [[], [], [], []]
                weight = [[], [], [], []]
                images = [image_list[0], image_list[1], image_list[2], image_list[3]]
                masks = [mask_list[0], mask_list[1], mask_list[2], mask_list[3]]
                for loader_model_index in range(4):
                    model = self.sub_model_list[loader_model_index]
                    img_1 = images[loader_model_index]
                    pred_1 = model(img_1)
                    mk_1 = masks[loader_model_index]
                    pred_1 = F.interpolate(pred_1, size=mk_1.shape[1:], mode='bilinear', align_corners=True)[:, 0,
                             :, :]
                    lce_1, ldice_1 = get_bce_dice_per_sample(pred_1, mk_1)
                    loss_ce[loader_model_index].extend(lce_1)
                    loss_dice[loader_model_index].extend(ldice_1)
                    
#                 total_loss, len_loss = [], []
#                 for index in range(4):
#                     print(f'len of loss_ce{index}:', len(loss_ce[index]))
#                     total_loss.extend(loss_ce[index])
#                     len_loss.append(len(loss_ce[index]))
                    
#                 total_loss = torch.stack(total_loss, dim=0)
#                 mean = torch.mean(total_loss) 

                for index in range(4):
                    loss_ce[index] = torch.stack(loss_ce[index], dim=0)
                    loss_ce[index] = torch.mean(loss_ce[index])
                    loss_dice[index] = torch.stack(loss_dice[index], dim=0)
                    loss_dice[index] = torch.mean(loss_dice[index])

                weight_list = []
                for weight_index in range(len(self.set_batch_size)):
                    weight_list.append(self.set_batch_size[weight_index] * 1.0 / self.total_batch_size)

                losses = torch.stack([loss_ce[0], loss_ce[1], loss_ce[2], loss_ce[3]], dim=0)
                mean = torch.mean(losses)
                
                # print(StopIteration_mark)
                # if len_loss[0] == 37 and len_loss[1] == 7 and len_loss[2] == 34 and len_loss[3] == 9:
                #     penalty = ((losses - mean) ** 2).mean()
                # else:
                #     penalty = 0
                
                penalty = ((losses - mean) ** 2).mean()

                # total_loss = (loss_ce[0] + loss_dice[0]) * weight_list[0] \
                #              + (loss_ce[1] + loss_dice[1]) * weight_list[1] \
                #              + (loss_ce[2] + loss_dice[2]) * weight_list[2] \
                #              + (loss_ce[3] + loss_dice[3]) * weight_list[3] \
                #              + penalty * self.args.penalty_weight
                total_loss = (loss_ce[0] + loss_dice[0]) \
                             + (loss_ce[1] + loss_dice[1]) \
                             + (loss_ce[2] + loss_dice[2]) \
                             + (loss_ce[3] + loss_dice[3]) \
                             + penalty * self.args.penalty_weight
                for index in range(4):
                    self.logger_list[index].add_scalar('ood/ce_loss', loss_ce[index], global_step)
                    self.logger_list[index].add_scalar('ood/dice_loss', loss_dice[index], global_step)
                    self.logger_list[index].add_scalar('ood/penalty', penalty, global_step)
                for opt_index in range(self.args.site_num):
                    self.optimizer_list[opt_index].zero_grad()
                total_loss.backward()
                for opt_index in range(self.args.site_num):
                    self.optimizer_list[opt_index].step()
                global_step += 1
                print(global_step)

            print('[Epoch %d/%d]%s' % (
                epoch + 1, self.args.epoch, datetime.datetime.now()))
            for loader_model_index in range(self.args.site_num):
                if epoch + 1 > self.args.start_test_epoch:
                    cur_model_path = self.args.savepath + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_model_path):
                        os.makedirs(cur_model_path)
                    RunFedTestPrivateBaseline(self.sub_model_list[loader_model_index],
                                                 self.args, cur_model_path, epoch)


                if epoch + 1 > self.args.start_save_model and (epoch + 1) % self.args.save_model_per_epoch == 0:
                    cur_path = self.args.savepath + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_path):
                        os.makedirs(cur_path)
                    # torch.save(self.sub_model_list[loader_model_index].state_dict(),
                    #            self.args.savepath + self.local_time + '/model/Client' + str(
                    #                loader_model_index + 1) + '/model-epoch=' + str(epoch + 1))

            
            com_counter += 1

        log_file = open(self.args.savepath + self.local_time + '/args.txt', "w")
        for arg in vars(self.args):
            log_file.write(str(arg) + ':' + str(getattr(self.args, arg)) + '\n')
        for site in range(self.args.site_num):
            print("[FedAvg+OOD] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
            log_file.write("[FedAvg+OOD] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
        for ret_index in range(len(self.ret_0_list)):
            log_file.write("[ret_0]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_0_list[ret_index]) + "\n")
            log_file.write("[ret_1]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_1_list[ret_index]) + "\n")

        log_file.close()
