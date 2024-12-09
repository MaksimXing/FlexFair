"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import pandas as pd
import os
from Code.F_SkinCancerClassification.dataset.BCN_dataset_input import BCN_SkinDataset, GetBCNDF
from Code.F_SkinCancerClassification.dataset.HAM_dataset_input import HAM_SkinDataset, GetHAMDF
from torchvision import transforms
from sklearn.metrics import average_precision_score


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def HAM_data_preparation(args, kwargs):
    base_skin_dir = os.path.join(args.datapath, 'Site', 'HAM10000')
    HAM_train_df = GetHAMDF(base_skin_dir, 'HAM10000.csv')

    validation_skin_dir = os.path.join(args.datapath, 'Test')
    HAM_validation_df = GetHAMDF(validation_skin_dir, 'metadata_no_blank.csv')
    ###### DataSet ######
    composed = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.CenterCrop(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    training_set = HAM_SkinDataset(HAM_train_df, transform=composed)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation_set = HAM_SkinDataset(HAM_validation_df, transform=composed)
    validation_generator = torch.utils.data.SequentialSampler(validation_set)
    return training_generator, validation_set, validation_generator


def BCN_data_preparation(args, kwargs):
    base_skin_dir = os.path.join(args.datapath, 'Site', 'BCN20000')
    BCN_train_df = GetBCNDF(base_skin_dir, 'BCN20000.csv')

    validation_skin_dir = os.path.join(args.datapath, 'Test')
    BCN_validation_df = GetBCNDF(validation_skin_dir, 'metadata_no_blank.csv')
    ###### DataSet ######
    composed = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.CenterCrop(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    training_set = BCN_SkinDataset(BCN_train_df, transform=composed)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation_set = BCN_SkinDataset(BCN_validation_df, transform=composed)
    validation_generator = torch.utils.data.SequentialSampler(validation_set)
    return training_generator, validation_set, validation_generator


def mkdiry(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_eo(result_list, gt_array, A_list):
    # Filter for the positive class
    positive_indices = gt_array == 1
    result_list_positive = result_list[positive_indices]
    A_list_positive = A_list[positive_indices]

    # Calculate the overall mean prediction for the positive class
    overall_mean_positive = np.mean(result_list_positive)

    unique_a = np.unique(A_list_positive)
    max_gap = 0  # Initialize the maximum gap

    for a in unique_a:
        # Indices for each group within the positive class
        group_indices = A_list_positive == a
        group_mean = np.mean(result_list_positive[group_indices])  # Mean prediction for the group

        # Calculate the absolute gap
        gap = abs(group_mean - overall_mean_positive)
        max_gap = max(max_gap, gap)  # Update the maximum gap if current gap is larger

    return max_gap



def calculate_dp(result_list, A_list):

    unique_a = np.unique(A_list)
    overall_mean = np.mean(result_list)  # Calculate the mean prediction for the entire dataset
    print('result_list.shape:', result_list.shape)
    print('result_list:', result_list)
    print('overall_mean:', overall_mean)
    
    max_gap = 0  # Initialize the maximum gap

    for a in unique_a:
        idx = A_list == a
        group_mean = np.mean(result_list[idx])  # Calculate mean prediction for each group
        print('group_mean:', group_mean)
        gap = abs(group_mean - overall_mean)  # Calculate the absolute gap
        max_gap = max(max_gap, gap)  # Update the maximum gap if current gap is larger

    return max_gap

def FedTest(train_loader, validation_set, model, epoch, writer):
    dataset_size = []
    dataset_total_size = 0
    for index in range(2):
        dataset_size.append(len(validation_set[index]))
        dataset_total_size += len(validation_set[index])

    # 0 and 1 are same('\Test'), so we only use 0
    accuracy, ap, Site_dp_results, Site_eo_results, Sex_dp_results, Sex_eo_results, Age_dp_results, Age_eo_results, csv_info = \
        test(train_loader[0], validation_set[0], model, epoch)
    
    writer.writerow({'epoch': epoch, 'acc':accuracy, 'ap': ap,
                     'Site_DP': Site_dp_results,
                     'Site_EO': Site_eo_results,
                     'Age_DP': Age_dp_results,
                     'Age_EO': Age_eo_results,
                     'Sex_DP': Sex_dp_results,
                     'Sex_EO': Sex_eo_results,})

    csv_data = {
        "path": csv_info[0],
        "result": csv_info[1],
        "gt": csv_info[2],
        "A_sex": csv_info[3],
        "A_age": csv_info[4],
        "A_site": csv_info[5]
    }

    df = pd.DataFrame(csv_data)

    print('Epoch: {:d} acc: {:.4f}  ap: {:.4f} SiteDP: {:.4f} AgeDP: {:.4f} SexDP: {:.4f}'.format(epoch + 1, accuracy, ap, Site_dp_results, Age_dp_results, Sex_dp_results))
    
    return df

def test(train_loader, validation_set, model, epoch):
    model.eval()
    result_array = []
    pred_array = []
    gt_array = []
    A_sex_array = []
    A_age_array = []
    A_site_array = []
    path_array = []
    y_scores = []
    y_true = []
    
    # for i in train_loader:
    for i in train_loader:
    
        data_sample, y, sex, age, site, path = validation_set.__getitem__(i)
        data_gpu = data_sample.unsqueeze(0).cuda()
        output = model(data_gpu)
        pred_array.append(output.item())

        result = (output >= 0.5).to(torch.int)
        result_array.append(result.item())
        
        # result = torch.argmax(output)
        # result_array.append(result.item())
        
        gt_array.append(y.item())
        A_sex_array.append(sex)
        A_age_array.append(age)
        A_site_array.append(site)
        path_array.append(path)


    result_array = np.array(result_array)
    pred_array = np.array(pred_array)
    gt_array = np.array(gt_array)
    A_sex_array = np.array(A_sex_array)
    A_age_array = np.array(A_age_array)
    A_site_array = np.array(A_site_array)
    path_array = np.array(path_array)


    Site_eo_results = calculate_eo(pred_array, gt_array, A_site_array)
    Site_dp_results = calculate_dp(pred_array, A_site_array)
    Age_eo_results = calculate_eo(pred_array, gt_array, A_age_array)
    Age_dp_results = calculate_dp(pred_array, A_age_array)
    Sex_eo_results = calculate_eo(pred_array, gt_array, A_sex_array)
    Sex_dp_results = calculate_dp(pred_array, A_sex_array) 
    
    correct_results = np.array(result_array) == np.array(gt_array)
    sum_correct = np.sum(correct_results)
    accuracy = sum_correct / train_loader.__len__()
    
    ap = average_precision_score(gt_array, pred_array)
    
    return accuracy, ap, Site_dp_results, Site_eo_results, Sex_dp_results, Sex_eo_results, Age_dp_results, Age_eo_results,\
[path_array, pred_array, gt_array, A_sex_array, A_age_array, A_site_array]


def test_client(train_loader, validation_set, model, epoch):
    model.eval()
    result_array = []
    pred_array = []
    gt_array = []
    A_sex_array = []
    A_age_array = []
    A_site_array = []
    path_array = []
    y_scores = []
    y_true = []
    
    # for i in train_loader:
    for i in train_loader:
    
        data_sample, y, sex, age, site, path = validation_set.__getitem__(i)
        data_gpu = data_sample.unsqueeze(0).cuda()
        output = model(data_gpu)
        pred_array.append(output.item())

        result = (output >= 0.5).to(torch.int)
        result_array.append(result.item())
        
        # result = torch.argmax(output)
        # result_array.append(result.item())
        
        gt_array.append(y.item())
        A_sex_array.append(sex)
        A_age_array.append(age)
        A_site_array.append(site)
        path_array.append(path)


    result_array = np.array(result_array)
    pred_array = np.array(pred_array)
    gt_array = np.array(gt_array)
    A_sex_array = np.array(A_sex_array)
    A_age_array = np.array(A_age_array)
    A_site_array = np.array(A_site_array)
    path_array = np.array(path_array)


    Site_eo_results = calculate_eo(pred_array, gt_array, A_site_array)
    Site_dp_results = calculate_dp(pred_array, A_site_array)
    Age_eo_results = calculate_eo(pred_array, gt_array, A_age_array)
    Age_dp_results = calculate_dp(pred_array, A_age_array)
    Sex_eo_results = calculate_eo(pred_array, gt_array, A_sex_array)
    Sex_dp_results = calculate_dp(pred_array, A_sex_array) 
    
    # site_age_dp
    site0_age_dp = calculate_dp(pred_array[A_site_array == 0], A_age_array[A_site_array == 0])
    site1_age_dp = calculate_dp(pred_array[A_site_array == 1], A_age_array[A_site_array == 1])
    site0_sex_dp = calculate_dp(pred_array[A_site_array == 0], A_sex_array[A_site_array == 0])
    site1_sex_dp = calculate_dp(pred_array[A_site_array == 1], A_sex_array[A_site_array == 1])   
    
    site0_age_eo = calculate_eo(pred_array[A_site_array == 0], gt_array[A_site_array == 0], A_age_array[A_site_array == 0])
    site1_age_eo = calculate_eo(pred_array[A_site_array == 1], gt_array[A_site_array == 1], A_age_array[A_site_array == 1])
    site0_sex_eo = calculate_eo(pred_array[A_site_array == 0], gt_array[A_site_array == 0], A_sex_array[A_site_array == 0])
    site1_sex_eo = calculate_eo(pred_array[A_site_array == 1], gt_array[A_site_array == 1], A_sex_array[A_site_array == 1])       
    
    correct_results = np.array(result_array) == np.array(gt_array)
    sum_correct = np.sum(correct_results)
    accuracy = sum_correct / train_loader.__len__()
    
    ap = average_precision_score(gt_array, pred_array)
    
    return site0_age_dp, site1_age_dp, site0_sex_dp, site1_sex_dp, site0_age_eo, site1_age_eo, site0_sex_eo, site1_sex_eo



def FairFedTest(train_loader, validation_set, model, epoch, writer, model_type):
    dataset_size = []
    dataset_total_size = 0
    for index in range(2):
        dataset_size.append(len(validation_set[index]))
        dataset_total_size += len(validation_set[index])

    if model_type == 'global':
        accuracy, ap, Site_dp_results, Site_eo_results, Sex_dp_results, Sex_eo_results, Age_dp_results, Age_eo_results, csv_info = \
            test(train_loader[0], validation_set[0], model, epoch)

        writer.writerow({'epoch': epoch, 'acc':accuracy, 'ap': ap,
                         'Site_DP': Site_dp_results,
                         'Site_EO': Site_eo_results,
                         'Age_DP': Age_dp_results,
                         'Age_EO': Age_eo_results,
                         'Sex_DP': Sex_dp_results,
                         'Sex_EO': Sex_eo_results,})

        csv_data = {
            "path": csv_info[0],
            "result": csv_info[1],
            "gt": csv_info[2],
            "A_sex": csv_info[3],
            "A_age": csv_info[4],
            "A_site": csv_info[5]
        }

        df = pd.DataFrame(csv_data)

        print('Epoch: {:d} acc: {:.4f}  ap: {:.4f} SiteDP: {:.4f} AgeDP: {:.4f} SexDP: {:.4f}'.format(epoch + 1, accuracy, ap, Site_dp_results, Age_dp_results, Sex_dp_results))

        return df, Sex_dp_results, Sex_eo_results, Age_dp_results, Age_eo_results
        
        
    elif model_type == 'client':
        site0_age_dp, site1_age_dp, site0_sex_dp, site1_sex_dp, site0_age_eo, site1_age_eo, site0_sex_eo, site1_sex_eo = \
            test_client(train_loader[0], validation_set[0], model, epoch)
        
        Age_dp_results = [site0_age_dp, site1_age_dp]
        Age_eo_results = [site0_age_eo, site1_age_eo]
        Sex_dp_results = [site0_sex_dp, site1_sex_dp]
        Sex_eo_results = [site0_sex_eo, site1_sex_eo]
        
        df = None
        
        return df, Sex_dp_results, Sex_eo_results, Age_dp_results, Age_eo_results