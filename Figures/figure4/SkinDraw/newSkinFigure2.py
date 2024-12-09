import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from natsort import natsorted

valve_ = [0.73, 0.73, 0.735, 0.735] # filter low dice result

# set fairness target
datasets = ['Age_DP', 'Age_EO', 'Sex_DP', 'Sex_EO']  
dataset_label = ['Age DP', 'Age EO', 'Gender DP', 'Gender EO'] 


labels = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova', 'FairMixup', 'FairFed', 'FlexFair']
ood_founder_dict = {'Age_DP': 'age-dp', 'Age_EO': 'age-eo', 'Sex_DP': 'sex-dp', 'Sex_EO': 'sex-eo'}
mixup_founder_dict = {'Age_DP': 'age-mixup', 'Age_EO': 'age-mixup', 'Sex_DP': 'sex-mixup', 'Sex_EO': 'sex-mixup'}

ylim_range = [[0.10, 0.14], [0.03, 0.06], [0.005, 0.02], [0.02, 0.040]]

# 遍历每个数据集
for idx_data, dataset in enumerate(datasets):
    methods_scores = {label: {'best_mean': [], 'min_mean': []} for label in labels}
    best_model_path = ''
    best_model_dice = 0
    
    for seed in range(5):

        step2_marker = False

        FedAvg_paths = f'your_data_path\\FedAvg\\seed{seed}'
        FedNova_paths = f'your_data_path\\FedNova\\seed{seed}\\'
        FedProx_paths = f'your_data_path\\FedProx\\seed{seed}\\'
        Scaffold_paths = f'your_data_path\\Scaffold\\seed{seed}\\'

        ood_df_paths = f'your_data_path\\{ood_founder_dict[dataset]}-scaff-ap-unweighted\\seed{seed}\\'

        mixup_df_paths = f'your_data_path\\{mixup_founder_dict[dataset]}\\seed{seed}\\'

        fairfed_df_paths = f'your_data_path\\{ood_founder_dict[dataset]}-fairfed\\seed{seed}\\'

        
        weighted_paths = [FedNova_paths, FedProx_paths, Scaffold_paths, ood_df_paths, mixup_df_paths, fairfed_df_paths]
        weighted_labels = ["FedNova", "FedProx", "SCAFFOLD", "FlexFair", "FairMixup", "FairFed"]
        for idx_, path_ in enumerate(weighted_paths):
            best_total_mean_ood = -np.inf
            min_total_gap_ood = np.inf
            min_gap_total_mean_ood = 0
            best_total_mean_gap = -np.inf

            paths = natsorted(glob.glob(os.path.join(path_, '*')))  # sort path

            for idx, base_df_path in paths:
                # fit paths
                if idx_ != 5:
                    global_file = os.path.join(glob.glob(os.path.join(base_df_path, '*'))[0],'result.csv')
                else:
                    global_file = os.path.join(base_df_path, 'result.csv')

                if os.path.exists(global_file):
                    df_global = pd.read_csv(global_file)
                    dice_values = df_global['ap'].values
                    max_dice = np.max(dice_values)  # get Max Dice
                    max_gap_each_epoch = df_global[dataset].values

                    # get Max Dice
                    if max_dice > best_total_mean_ood and max_dice > valve_[idx_data]:
                        best_total_mean_ood = max_dice
                        best_total_mean_gap = max_gap_each_epoch[np.argmax(dice_values)]
                    # Select base model for FlexFair
                    if max_dice > best_model_dice and idx_ == 3:
                        best_model_dice = max_dice
                        best_model_path = global_file

            if best_total_mean_gap != -np.inf:
                methods_scores[weighted_labels[idx_]]['best_mean'].append(best_total_mean_gap)
                if idx_ == 4:
                    step2_marker = True

        # Deal with FedAvg
        best_total_mean_base = -np.inf
        min_total_gap_base = np.inf
        min_gap_total_mean_base = 0
        best_total_mean_gap = -np.inf

        global_file = os.path.join(glob.glob(os.path.join(FedAvg_paths, '*'))[0],'result.csv')
        if os.path.exists(global_file):
            df_global = pd.read_csv(global_file)
            dice_values = df_global['ap'].values
            max_dice = np.max(dice_values)  # get Max Dice
            max_gap_each_epoch = df_global[dataset].values

            # get Max Dice
            if max_dice > best_total_mean_base and max_dice > valve_[idx_data]:
                best_total_mean_base = max_dice
                best_total_mean_gap = max_gap_each_epoch[np.argmax(dice_values)]

        if best_total_mean_gap != -np.inf:
            methods_scores['FedAvg']['best_mean'].append(best_total_mean_gap)
            if step2_marker:
                pop_ = methods_scores['FairMixup']['best_mean'].pop()
                if best_total_mean_gap >  pop_:
                    methods_scores['FairMixup']['best_mean'].append(best_total_mean_gap)
                else:
                    methods_scores['FairMixup']['best_mean'].append(pop_)
            else:
                methods_scores['FairMixup']['best_mean'].append(best_total_mean_gap)

    
    fig, ax1 = plt.subplots(figsize=(4, 3))

    # set bar width
    width = 0.35
    x = np.arange(len(labels))  

    # generate color
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels) + 1))  

    
    ax1.bar(x, [np.mean(methods_scores[label]['best_mean']) for label in labels], width,
            yerr=[np.std(methods_scores[label]['best_mean']) for label in labels], capsize=5,
            color=colors[1:])

    # Add Info
    ax1.set_ylim(ylim_range[idx_data][0], ylim_range[idx_data][1])
    ax1.set_xlabel('Methods')
    ax1.set_ylabel(dataset_label[idx_data])
    # ax1.set_title('MaxGap for Min MaxGap')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    # ax1.legend(loc='upper right')
    # ax1.set_ylim(0.85, 0.9)
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)

    plt.tight_layout()
    plt.savefig('Skin_' + dataset + '-Fig2.pdf')
    print(best_model_path)
    print(best_model_dice)
    plt.show()
