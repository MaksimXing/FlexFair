from util import BCN_data_preparation, HAM_data_preparation, mkdiry, FedTest
# import torchvision.models as models
from method import resnet
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tqdm
import torch
from datetime import datetime
import copy
import csv

def ScaffoldTrain(args, kwargs):
    com_counter = 0

    now = datetime.now()
    local_time = now.strftime("%m%d-%H%M%S")
    path = f'output/Scaffold-2/s{args.seed}_w{args.c_lr}/' + local_time
    
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

    # scaffold model setting
    c_global_model = resnet.resnet50(pretrained=True)
    c_global_model.fc = nn.Linear(in_features=2048, out_features=1)
    c_global_model.cuda()
    c_global_para = c_global_model.state_dict()
    c_sub_model_list = []
    for site in range(2):
        c_temp_model = resnet.resnet50(pretrained=True)
        c_temp_model.fc = nn.Linear(in_features=2048, out_features=1)
        c_temp_model.cuda()
        c_sub_model_list.append(c_temp_model)
    # replace client with global(for init)
    for c_model_index in range(2):
        c_sub_model_list[c_model_index].load_state_dict(c_global_para)

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

    # init
    c_global_para = c_global_model.state_dict()
    c_local_para_list = []
    total_delta = copy.deepcopy(global_model.state_dict())
    cnt = list(0 for i in range(2))

    for epoch in tqdm.tqdm(range(args.epochs)):

        # Fed
        global_para = global_model.state_dict()
        # Init
        if com_counter == 0:
            # replace client with global(for init)
            for index in range(2):
                model_list[index].load_state_dict(global_para)
            # set new total_data
            total_delta = copy.deepcopy(global_model.state_dict())
            for key in total_delta:
                total_delta[key] = 0.0
            c_global_para = c_global_model.state_dict()
            c_local_para_list = []
            for c_model_index in range(2):
                c_local_para_list.append(c_sub_model_list[c_model_index].state_dict())
            # cnt = 0
            cnt = list(0 for i in range(2))
        elif com_counter % args.com_round == 0:
            print("*** Scaffold Execute ***")
            # update local c_i
            for c_model_index in range(2):
                c_new_para = c_sub_model_list[c_model_index].state_dict()
                c_delta_para = copy.deepcopy(c_sub_model_list[c_model_index].state_dict())
                global_model_para = global_model.state_dict()
                net_para = model_list[c_model_index].state_dict()
                for key in net_para:
                    # c_new_para[key] and c_global_para[key] are on CPU
                    # net_para[key] and global_model_para[key] are on CUDA
                    c_new_para[key] = c_new_para[key].cuda()
                    c_global_para[key] = c_global_para[key].cuda()
                    net_para[key] = net_para[key].cuda()
                    global_model_para[key] = global_model_para[key].cuda()
                    c_new_para[key] = c_new_para[key] - c_global_para[key] \
                                      + (global_model_para[key] - net_para[key]) / (cnt[c_model_index] * float(args.c_lr))
                    # c_new_para[key].device is on CUDA:0
                    # c_local_para_list[c_model_index][key] and c_delta_para[key] is on CPU
                    c_new_para[key] = c_new_para[key].cpu()
                    c_delta_para[key] = c_new_para[key] - c_local_para_list[c_model_index][key]
                c_sub_model_list[c_model_index].load_state_dict(c_new_para)
                # sum updated C
                for key in total_delta:
                    total_delta[key] += c_delta_para[key]
            # for all clients
            # sum updated C and get avg
            for key in total_delta:
                total_delta[key] /= 2
            c_global_para = c_global_model.state_dict()
            for key in c_global_para:
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    c_global_para[key] = c_global_para[key].cpu()
                    c_global_para[key] += total_delta[key]
            c_global_model.load_state_dict(c_global_para)

            for index in range(2):
                sub_net_para = model_list[index].state_dict()
                # Calculate Federated
                if index == 0:
                    for key in sub_net_para:
                        global_para[key] = sub_net_para[key] \
                                           * (dataset_size[index] * 1.0 / dataset_total_size)
                else:
                    for key in sub_net_para:
                        global_para[key] += sub_net_para[key] \
                                            * (dataset_size[index] * 1.0 / dataset_total_size)
            # update global
            global_model.load_state_dict(global_para)
            # evaluate on validation set
            df = FedTest(validation_generator_list, validation_set_list, global_model, epoch, writer)
            mkdiry(path + '/model/')
            df.to_csv(path + '/model/[EPOCH' + str(epoch) + '.csv', index=False)
            torch.save(global_model.state_dict(),
                       path + '/model/[EPOCH' + str(epoch) + ']global-model.pth')
            # update client
            for index in range(2):
                model_list[index].load_state_dict(global_para)

            # reset new total_data
            total_delta = copy.deepcopy(global_model.state_dict())
            for key in total_delta:
                total_delta[key] = 0.0
            c_global_para = c_global_model.state_dict()
            c_local_para_list = []
            for c_model_index in range(2):
                c_local_para_list.append(c_sub_model_list[c_model_index].state_dict())
            # cnt = 0
            cnt = list(0 for i in range(2))

            print("*** Scaffold Complete ***")

            
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

            loss_ce = [[], []]
            loss_ce_EO = [[], []]
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

            loss = (loss_site[0]) + (loss_site[1])

            ## backward
            for index in range(2):
                optimizer_list[index].zero_grad()
            loss.backward()

            for index in range(2):
                optimizer_list[index].step()
                
                # send para
                # init as $y_i \leftarrow x$
                net_para = model_list[index].state_dict()
                c_lr = optimizer_list[index].param_groups[0]['lr']

                for key in net_para:
                    # c_global_para[key]和c_local_para_list[loader_model_index][key]on cpu
                    # net_para[key]on cuda
                    net_para[key] = net_para[key].cpu()
                    c_local_para_list[index][key] = c_local_para_list[index][key].cpu()
                    c_global_para[key] = c_global_para[key].cpu()
                    net_para[key] = net_para[key] - c_lr * (c_global_para[key] - c_local_para_list[index][key])
                model_list[index].load_state_dict(net_para)

                cnt[index] += 1
                
        for index in range(2):
            lr_scheduler_list[index].step()

        com_counter += 1
          
            
            
            
        

