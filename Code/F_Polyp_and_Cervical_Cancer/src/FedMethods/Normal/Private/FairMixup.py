import os

from FedMethods.FedRequirements import *
# from FedBasicFunc.Models.FedAvgModel import FedAvgModel
from FedBasicFunc.Models.FairMixupModel import FairMixupModel
from numpy.random import beta

class FairMixupTrainPrivate(object):
    def __init__(self, DataList, FairMixupModel, args):
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
        self.global_model = FairMixupModel(args)
        self.global_model.train(True)
        self.global_model.cuda()

        model_state_dict = torch.load(f'../res/private/avg_seed{args.seed}_model')
        self.global_model.load_state_dict(model_state_dict, strict = True)
        
        
        ## SUB model
        self.sub_model_list = []
        for site in range(args.site_num):
            self.sub_model_list.append(FairMixupModel(args))
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
                SummaryWriter(args.savepath + '/FairMixup/' + self.local_time + '/log/Client' + str(model_index + 1)))

    def train(self):
        global_step_list = list(0 for i in range(self.args.site_num))

        # Fed counter
        com_counter = 1

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

            # Mixup
            alpha = 1
            gamma = beta(alpha, alpha)

            # init iter
            data_iter0 = iter(self.loader_list[0])
            data_iter1 = iter(self.loader_list[1])
            data_iter2 = iter(self.loader_list[2])
            
            for loader_model_index in range(self.args.site_num):
                epoch_start_time = time.process_time()

                loss_ce = 0
                loss_dice = 0
                for image, mask in self.loader_list[loader_model_index]:
                    
                    try:
                        # get loader data(batch)
                        loss_reg = 0
                        batch_x0, batch_y0 = next(data_iter0)
                        batch_x1, batch_y1 = next(data_iter1)
                        batch_x2, batch_y2 = next(data_iter2)
                        
                        batch_x0, batch_y0 = batch_x0.cuda(), batch_y0.cuda()
                        batch_x1, batch_y1 = batch_x1.cuda(), batch_y1.cuda()
                        batch_x2, batch_y2 = batch_x2.cuda(), batch_y2.cuda()
                        
                        batch_x_mix = [
                            batch_x0 * gamma + batch_x1 * (1 - gamma),
                            batch_x1 * gamma + batch_x2 * (1 - gamma),
                            batch_x2 * gamma + batch_x3 * (1 - gamma),
                        ]
                        
                        # reg on grad
                        batch_x = batch_x_mix[loader_model_index]
                        # reg loss
                        # print(batch_x.shape)
                        batch_x = batch_x.requires_grad_(True)
                        output = self.model_list[loader_model_index](batch_x)
                        # print(output.sum())
                        gradx = torch.autograd.grad(output.sum(), batch_x, create_graph=True)[0].view(batch_x.shape[0], -1)
                        batch_x_d = batch_x - [batch_x0, batch_x1, batch_x2][loader_model_index]
                        batch_x_d = batch_x_d.view(batch_x.shape[0], -1)
                        # print('shape', (gradx * batch_x_d).shape)
                        grad_inn = (gradx * batch_x_d).sum(1).view(-1)
                        E_grad = grad_inn.mean()
                        loss_reg += torch.abs(E_grad)
                        # loss_reg = loss_reg.mean()

                    except:
                        loss_reg += 0
                    
                    
                    single_image_start_time = time.process_time()
                    image, mask = image.cuda().float(), mask.cuda().float()

                    rand = int(np.random.choice([256, 288, 320], p=[0.2, 0.3, 0.5]))
                    image = F.interpolate(image, size=(rand, rand), mode='bilinear')
                    mask = F.interpolate(mask.unsqueeze(1), size=(rand, rand), mode='nearest').squeeze(1)

                    pred = self.sub_model_list[loader_model_index](image)
                    pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)[:, 0, :, :]
                    loss_ce, loss_dice = bce_dice(pred, mask)

                    self.optimizer_list[loader_model_index].zero_grad()
                    scale_loss = loss_ce + loss_dice + loss_reg * self.args.penalty_weight
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
                    cur_model_path = self.args.savepath + '/FairMixup/' + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_model_path):
                        os.makedirs(cur_model_path)
                    RunFedTestPrivateBaseline(self.sub_model_list[loader_model_index],
                                                     self.args, cur_model_path, epoch)


                if epoch + 1 > self.args.start_save_model and (epoch + 1) % self.args.save_model_per_epoch == 0:
                    cur_path = self.args.savepath + '/FairMixup/' + self.local_time + '/model/Client' + str(
                        loader_model_index + 1) + '/'
                    if not os.path.exists(cur_path):
                        os.makedirs(cur_path)
                    # torch.save(self.sub_model_list[loader_model_index].state_dict(),
                    #            self.args.savepath + '/FairMixup/' + self.local_time + '/model/Client' + str(
                    #                loader_model_index + 1) + '/model-epoch=' + str(epoch + 1))

                epoch_end_time = time.process_time()
                self.logger_list[loader_model_index].add_scalar('time/epoch_process',
                                                                epoch_end_time - epoch_start_time,
                                                                epoch + 1)
            # for every epoch, every sub model run once
            if com_counter != 0 and com_counter % self.args.com_round == 0:
                # calculate communicate cost
                sub_net_size = 0
                for model_index in range(self.args.site_num):
                    sub_net_size += sys.getsizeof(self.sub_model_list[model_index].state_dict())
                print("*** FairMixup Execute ***")
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
                cur_model_path = self.args.savepath + '/FairMixup/' + self.local_time + '/model/'
                if not os.path.exists(cur_model_path):
                    os.makedirs(cur_model_path)
                # torch.save(self.global_model.state_dict(), cur_model_path + 'model_global-epoch=' + str(epoch + 1))
                RunFedTestPrivateBaseline(self.global_model,
                                                 self.args, cur_model_path, epoch)
                print("*** FairMixup Complete ***")
            
            com_counter += 1

        log_file = open(self.args.savepath + '/FairMixup/' + self.local_time + '/args.txt', "w")
        for arg in vars(self.args):
            log_file.write(str(arg) + ':' + str(getattr(self.args, arg)) + '\n')
        for site in range(self.args.site_num):
            print("[FairMixup] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
            log_file.write("[FairMixup] SITE[" + str(site) + "]\tDICE: " + str(self.max_dice_test[site]))
        for ret_index in range(len(self.ret_0_list)):
            log_file.write("[ret_0]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_0_list[ret_index]) + "\n")
            log_file.write("[ret_1]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_1_list[ret_index]) + "\n")
            log_file.write("[ret_2]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_2_list[ret_index]) + "\n")
            log_file.write("[ret_3]Epoch:" + str(ret_index + 1) + "\t" + str(self.ret_3_list[ret_index]) + "\n")
        log_file.close()
