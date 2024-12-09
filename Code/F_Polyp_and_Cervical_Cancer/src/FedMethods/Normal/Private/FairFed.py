import os

from FedMethods.FedRequirements import *
from FedBasicFunc.Models.FedAvgModel import FedAvgModel


class FairFedTrainPrivate(object):
    def __init__(self, DataList, FedAvgModel, args):
        ## dataset
        self.args = args
        self.DataList = DataList

        self.ret_0_list = []
        self.ret_1_list = []
        self.ret_2_list = []
        self.ret_3_list = []

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
                SummaryWriter(args.savepath + self.local_time + '/log/Client' + str(model_index + 1)))

    def train(self):

        global_step_list = list(0 for i in range(self.args.site_num))

        # Fed counter
        com_counter = 0

        # get dataset size
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
                    # replace client with global(for init)
                    for model_index in range(self.args.site_num):
                        self.sub_model_list[model_index].load_state_dict(global_para)
                        
                    ratio = [dataset_size[index] * 1.0 / dataset_total_size for index in range(0,4)]
                        
            # for every epoch, every sub model run once
            elif com_counter % self.args.com_round == 0:
                # calculate communicate cost
                sub_net_size = 0
                for model_index in range(self.args.site_num):
                    sub_net_size += sys.getsizeof(self.sub_model_list[model_index].state_dict())
                print("*** FairFed Execute ***")
                aggregation_start_time = time.process_time()

                for model_index in range(self.args.site_num):
                    sub_net_para = self.sub_model_list[model_index].state_dict()
                    # Calculate Federated
                    if model_index == 0:
                        for key in sub_net_para:
                            global_para[key] = sub_net_para[key] * ratio[model_index]
                    else:
                        for key in sub_net_para:
                            global_para[key] += sub_net_para[key] * ratio[model_index]
                            
                # update global
                self.global_model.load_state_dict(global_para)
                # update client
                for model_index in range(self.args.site_num):
                    self.sub_model_list[model_index].load_state_dict(global_para)

                aggregation_end_time = time.process_time()

                # calculate communicate cost
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
                cur_model_path = self.args.savepath + self.local_time + '/model/'
                if not os.path.exists(cur_model_path):
                    os.makedirs(cur_model_path)
                # torch.save(self.global_model.state_dict(), cur_model_path + 'model_global-epoch=' + str(epoch + 1))
                
                # evaluate global model
                mean_global_dice, global_dice = RunFedTestPrivateBaseline(self.global_model,
                                                 self.args, cur_model_path, epoch)
                
                # evaluate client model
                client_gaps = []
                for loader_model_index in range(self.args.site_num):
                    mean_client_dice, client_dice = RunFedTestPrivateBaseline_client(self.sub_model_list[loader_model_index],
                                                 self.args, cur_model_path, epoch)
                    client_gaps.append(abs(global_dice - mean_client_dice[loader_model_index]))
                
                mean_gap = sum(client_gaps) / len(client_gaps)
                
                client_gaps = [x - mean_gap for x in client_gaps]
                
                for i in range(len(ratio)):
                    ratio[i] = ratio[i] - self.args.beta * client_gaps[i]             
                sum_ratio = sum(ratio)
                ratio = [x / sum_ratio for x in ratio]                
                print('ratio:', ratio)
                print("*** FairFed Complete ***")

            for loader_model_index in range(self.args.site_num):
                epoch_start_time = time.process_time()

                loss_ce = 0
                loss_dice = 0
                for image, mask in self.loader_list[loader_model_index]:
                    single_image_start_time = time.process_time()
                    image, mask = image.cuda().float(), mask.cuda().float()

                    rand = int(np.random.choice([256, 288, 320], p=[0.2, 0.3, 0.5]))
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

                epoch_end_time = time.process_time()
                self.logger_list[loader_model_index].add_scalar('time/epoch_process',
                                                                epoch_end_time - epoch_start_time,
                                                                epoch + 1)

            
            com_counter += 1

        log_file = open(self.args.savepath + self.local_time + '/args.txt', "w")
        for arg in vars(self.args):
            log_file.write(str(arg) + ':' + str(getattr(self.args, arg)) + '\n')
        for site in range(self.args.site_num):
            print("[FairFed] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
            log_file.write("[FairFed] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
        for ret_index in range(len(self.ret_0_list)):
            log_file.write("[ret_0]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_0_list[ret_index]) + "\n")
            log_file.write("[ret_1]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_1_list[ret_index]) + "\n")
            log_file.write("[ret_2]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_2_list[ret_index]) + "\n")
            log_file.write("[ret_3]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_3_list[ret_index]) + "\n")
        log_file.close()
