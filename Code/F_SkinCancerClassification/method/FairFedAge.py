from util import HAM_data_preparation, BCN_data_preparation, mkdiry, FedTest, FairFedTest
# import torchvision.models as models
from method import resnet
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tqdm
import torch
from datetime import datetime
import csv


def FairFedAgeTrain(args, kwargs):
    com_counter = 0

    now = datetime.now()
    local_time = now.strftime("%m%d-%H%M%S")
    path = f'output-fairfed/{args.sex_age}-{args.dp_eo}/s{args.seed}_w{args.penalty_weight}/' + local_time

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
            ratio = [dataset_size[index] * 1.0 / dataset_total_size for index in range(2)]
                
        elif com_counter % args.com_round == 0:
            print("*** Weighted Execute ***")
            for index in range(2):
                sub_net_para = model_list[index].state_dict()
                # Calculate Federated
                if index == 0:
                    for key in sub_net_para:
                        global_para[key] = sub_net_para[key] * ratio[index]
                else:
                    for key in sub_net_para:
                        global_para[key] += sub_net_para[key] * ratio[index]
            # update global
            global_model.load_state_dict(global_para)
            
            # update client
            for index in range(2):
                model_list[index].load_state_dict(global_para)

            
            # evaluate global model
            df, Sex_dp_results, Sex_eo_results, Age_dp_results, Age_eo_results = FairFedTest(validation_generator_list, validation_set_list, global_model, epoch, writer, 'global')
            
            mkdiry(path + '/model/')
            df.to_csv(path + '/model/[EPOCH' + str(epoch) + '.csv', index=False)
            # torch.save(global_model.state_dict(),
            #            path + '/model/[EPOCH' + str(epoch) + ']global-model.pth')            
            
            # evaluate client model
            client_gaps = []
            for index in range(2):
                df, Sex_dp_client_results, Sex_eo_client_results, Age_dp_client_results, Age_eo_client_results = FairFedTest(validation_generator_list, validation_set_list, model_list[index], epoch, writer, 'client') 
                
                if args.dp_eo == 'dp':
                    client_gaps.append(abs(Age_dp_results - Age_dp_client_results[index]))
                elif args.dp_eo == 'eo':
                    client_gaps.append(abs(Age_eo_results - Age_eo_client_results[index]))
                    
                    
            mean_gap = sum(client_gaps) / len(client_gaps)
            client_gaps = [x - mean_gap for x in client_gaps]

            for i in range(len(ratio)):
                ratio[i] = ratio[i] - args.beta * client_gaps[i]             
            sum_ratio = sum(ratio)
            ratio = [x / sum_ratio for x in ratio]              
            print('ratio:', ratio)
            
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


            loss = (loss_site[0]) + (loss_site[1])

            ## backward
            for index in range(2):
                optimizer_list[index].zero_grad()
            loss.backward()

            for index in range(2):
                optimizer_list[index].step()
        for index in range(2):
            lr_scheduler_list[index].step()

        com_counter += 1
