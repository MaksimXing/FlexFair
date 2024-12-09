from util import HAM_data_preparation, BCN_data_preparation, mkdiry, FedTest
# import torchvision.models as models
from method import resnet
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tqdm
import torch
from datetime import datetime
import csv


def get_bce_dice_per_sample_EO(id_criterion, predictions, masks, sexs, ages):
    loss_ce_index_00 = []
    loss_ce_index_01 = []
    loss_ce_index_10 = []
    loss_ce_index_11 = []
    for output, y, sex, age in zip(predictions, masks, sexs, ages):
        output = output.cuda()
        y = y.cuda()
        
        # output = output.unsqueeze(0).cuda()  
        # y = y.unsqueeze(0).cuda()   
        
        l_ce = id_criterion(output, y)
        # judge attribute
        if y == 0 and sex == 0:
            loss_ce_index_00.append(l_ce)
        elif y == 0 and sex == 1:
            loss_ce_index_01.append(l_ce)
        elif y == 1 and sex == 0:
            loss_ce_index_10.append(l_ce)
        elif y == 1 and sex == 1:
            loss_ce_index_11.append(l_ce)
        else:
            return ArithmeticError
        
    return loss_ce_index_00, loss_ce_index_01, loss_ce_index_10, loss_ce_index_11


def NewWeightedSexTrain(args, kwargs):
    com_counter = 0

    now = datetime.now()
    local_time = now.strftime("%m%d-%H%M%S")
    path = f'output-flexfair-new/{args.sex_age}-{args.dp_eo}-scaff-ap/s{args.seed}_w{args.penalty_weight}/' + local_time
    
    mkdiry(path)

    # csv file
    filename = path + '/result.csv'
    headers = ['epoch', 'acc', 'ap', 'Site_DP', 'Site_EO', 'Age_DP', 'Age_EO', 'Sex_DP', 'Sex_EO']
    # open csv
    csvfile = open(filename, 'w', newline='', encoding='utf-8')
    # creat csv writer
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()

    # resize batch
    dataset_size = [6174, 4904]
    set_batch_size = []
    for size_ in dataset_size:
        set_batch_size.append(round(size_ * 1.0 / args.fairness_step))
    total_batch_size = sum(set_batch_size)
    args.batch_size = set_batch_size[0]
    HAM_training_generator, HAM_validation_set, HAM_validation_generator = HAM_data_preparation(args, kwargs)
    args.batch_size = set_batch_size[1]
    BCN_training_generator, BCN_validation_set, BCN_validation_generator = BCN_data_preparation(args, kwargs)

    training_generator_list = [HAM_training_generator, BCN_training_generator]
    validation_set_list = [HAM_validation_set, BCN_validation_set]
    validation_generator_list = [HAM_validation_generator, BCN_validation_generator]

    ###### Model ######
    model_list = []
    for index in range(2):
        model = resnet.resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=1)
        model.cuda()
        model_list.append(model)

    global_model = resnet.resnet50(pretrained=True)
    global_model.fc = nn.Linear(in_features=2048, out_features=1)
    global_model.cuda()

    cudnn.benchmark = True
    
    model_state_dict = torch.load(f'../models/scaff{args.seed}_ap_model.pth')
    global_model.load_state_dict(model_state_dict, strict = True)
    
    ###### Criteria ######
    # id_criterion = nn.CrossEntropyLoss()
    id_criterion = nn.BCELoss()
    optimizer_list = []
    lr_scheduler_list = []
    for index in range(2):
        optimizer_list.append(torch.optim.AdamW(model_list[index].parameters(), lr=args.lr))
        lr_scheduler_list.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[index], T_max=args.epochs, eta_min=0))

    dataset_size = []
    dataset_total_size = 0
    for loader in training_generator_list:
        dataset_size.append(len(loader.dataset))
        dataset_total_size += len(loader.dataset)

    for epoch in tqdm.tqdm(range(args.epochs)):

        # Fed
        global_para = global_model.state_dict()
        # Init
        if com_counter == 0:
            # replace client with global(for init)
            for index in range(2):
                model_list[index].load_state_dict(global_para)
        elif com_counter % args.com_round == 0:
            print("*** Weighted Execute ***")
            for index in range(2):
                sub_net_para = model_list[index].state_dict()
                # Calculate Federated
                if index == 0:
                    for key in sub_net_para:
                        global_para[key] = sub_net_para[key]\
                                           * (dataset_size[index] * 1.0 / dataset_total_size)
                else:
                    for key in sub_net_para:
                        global_para[key] += sub_net_para[key]\
                                           * (dataset_size[index] * 1.0 / dataset_total_size)
            # update global
            global_model.load_state_dict(global_para)
            # evaluate on validation set
            df = FedTest(validation_generator_list, validation_set_list, global_model, epoch, writer)
            mkdiry(path + '/model/')
            df.to_csv(path + '/model/[EPOCH' + str(epoch) + '.csv', index=False)
            # torch.save(global_model.state_dict(),
            #            path + '/model/[EPOCH' + str(epoch) + ']global-model.pth')
            
            # update client
            for index in range(2):
                model_list[index].load_state_dict(global_para)

            print("*** Weighted Complete ***")

        dataloader_iterator_list = []
        StopIteration_mark = []
        for loader_index in range(len(training_generator_list)):
            dataloader_iterator_list.append(iter(training_generator_list[loader_index]))
            StopIteration_mark.append(False)

        index_count = [0, 0]

        while True:
            data_sample_list = []
            y_list = []
            sex_list, age_list = [], []
            for loader_index in range(len(training_generator_list)):
                try:
                    data_sample, y, sex, age, site, path_ = next(dataloader_iterator_list[loader_index])
                    index_count[loader_index] += 1
                except StopIteration:
                    dataloader_iterator_list[loader_index] = iter(training_generator_list[loader_index])
                    data_sample, y, sex, age, site, path_ = next(dataloader_iterator_list[loader_index])
                    StopIteration_mark[loader_index] = True
                    index_count[loader_index] = 0

                # all iter complete
                if all(element == True for element in StopIteration_mark):
                    break

                data_sample, y = data_sample.cuda(), y.cuda()
                data_sample_list.append(data_sample)
                y_list.append(y)
                # read attribute
                sex_list.append(sex)
                age_list.append(age)

            # test again
            if all(element == True for element in StopIteration_mark):
                break

            loss_ce = [[], [], [], []]

            datas = [data_sample_list[0], data_sample_list[1]]
            ys = [y_list[0], y_list[1]]
            loss_site = []

            for loader_model_index in range(2):
                model = model_list[loader_model_index]
                model.train()
                img_1 = datas[loader_model_index]

                pred_1 = model(img_1)
                mk_1 = ys[loader_model_index]
                mk_1 = mk_1.unsqueeze(1).cuda().float()
                loss_site.append(id_criterion(pred_1, mk_1))

                loss_ce_index_00, loss_ce_index_01, loss_ce_index_10, loss_ce_index_11 = \
                    get_bce_dice_per_sample_EO(id_criterion, pred_1, mk_1,
                                               sex_list[loader_model_index], age_list[loader_model_index])
                loss_ce[0].extend(loss_ce_index_00)
                loss_ce[1].extend(loss_ce_index_01)
                loss_ce[2].extend(loss_ce_index_10)
                loss_ce[3].extend(loss_ce_index_11)


            weight_list = []
            for weight_index in range(len(set_batch_size)):
                weight_list.append(set_batch_size[weight_index] * 1.0 / total_batch_size)

            if args.dp_eo == 'dp':
                penalty = 0
                if len(loss_ce[0]) != 0 and len(loss_ce[1]) != 0:
                    total_loss = []
                    for index in range(2):
                        total_loss.extend(loss_ce[index])
                    total_loss = torch.stack(total_loss, dim=0)
                    mean = torch.mean(total_loss)
                    
                    for index in range(2):
                        loss_ce[index] = torch.stack(loss_ce[index], dim=0)
                        loss_ce[index] = torch.mean(loss_ce[index])
                        
                    # losses = torch.stack([loss_ce[0] * weight_list[0], loss_ce[1] * weight_list[1]], dim=0)
                    losses = torch.stack([loss_ce[0], loss_ce[1]], dim=0)

                    penalty += ((losses - mean) ** 2).mean()
                    
                if len(loss_ce[2]) != 0 and len(loss_ce[3]) != 0:
                    total_loss = []
                    for index in range(2, 4):
                        total_loss.extend(loss_ce[index])
                    total_loss = torch.stack(total_loss, dim=0)
                    mean = torch.mean(total_loss)
                    
                    for index in range(2, 4):
                        loss_ce[index] = torch.stack(loss_ce[index], dim=0)
                        loss_ce[index] = torch.mean(loss_ce[index])
                        
                    losses = torch.stack([loss_ce[2], loss_ce[3]], dim=0)

                    penalty += ((losses - mean) ** 2).mean()     
                    

            elif args.dp_eo == 'eo':
                if len(loss_ce[2]) != 0 and len(loss_ce[3]) != 0:
                    total_loss = []
                    for index in range(2, 4):
                        total_loss.extend(loss_ce[index])
                    total_loss = torch.stack(total_loss, dim=0)
                    mean = torch.mean(total_loss)
                    
                    for index in range(2, 4):
                        loss_ce[index] = torch.stack(loss_ce[index], dim=0)
                        loss_ce[index] = torch.mean(loss_ce[index])
                    # losses = torch.stack([loss_ce_EO[0] * weight_list[0], loss_ce_EO[1] * weight_list[1]], dim=0)
                    losses = torch.stack([loss_ce[2], loss_ce[3]], dim=0)

                    penalty = ((losses - mean) ** 2).mean()
                    
                else:
                    print('EO not compute this round')

                    penalty = 0
                    
            else:
                raise ValueError

            loss = (loss_site[0]) + (loss_site[1]) + penalty * float(args.penalty_weight)

            ## backward
            for index in range(2):
                optimizer_list[index].zero_grad()
            loss.backward()

            for index in range(2):
                optimizer_list[index].step()
        for index in range(2):
            lr_scheduler_list[index].step()
            
        com_counter += 1
