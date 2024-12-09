import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import matplotlib
from natsort import natsorted
matplotlib.use('TkAgg')


labels = ['FedAvg', 'FedProx', 'SCAFFOLD','FedNova', 'FairMixup', 'FairFed', 'FlexFair']
methods_scores = {label: {'best_mean': [], 'min_mean': []} for label in labels}


valve_ = 0.655 # filter low dice result

best_model_path = ''
best_model_dice = 0


for seed in range(5):

    step2_marker = False
    
    FedAvg_paths = f'your_data_path\\FedAvg\\seed{seed}\\'
    FedNova_paths = f'your_data_path\\FedNova\\seed{seed}\\'
    FedProx_paths = f'your_data_path\\FedProx\\seed{seed}\\'
    Scaffold_paths = f'your_data_path\\Scaffold\\seed{seed}\\'

    ood_df_paths = f'your_data_path\\FedAvgOOD\\seed{seed}\\' # FlexFair path

    fairfed_df_paths = f'your_data_path\\FairFed\\seed{seed}\\'

    mixup_df_paths = f'your_data_path\\FairMixup\\seed{seed}\\'

    
    weighted_paths = [FedNova_paths, FedProx_paths, Scaffold_paths, fairfed_df_paths,mixup_df_paths, ood_df_paths]
    weighted_labels = ["FedNova", "FedProx", "SCAFFOLD", "FairFed", "FairMixup", "FlexFair"]
    for idx_, path_ in enumerate(weighted_paths):
        
        best_total_mean_ood = -np.inf
        min_total_gap_ood = np.inf
        min_gap_total_mean_ood = 0
        best_total_mean_gap = -np.inf

        ood_df_paths = natsorted(glob.glob(os.path.join(path_, '*')))  # sort path
        
        for idx, base_df_path in ood_df_paths:
            global_file = os.path.join(base_df_path, 'result.csv')
            if os.path.exists(global_file):
                df_global = pd.read_csv(global_file, header=None)
                dice_values = df_global.iloc[:, 0].values
                max_dice = np.max(dice_values)  # get Max Dice

                # calculate MaxGap
                gap_column_2 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 2].values)
                gap_column_3 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 3].values)
                gap_column_4 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 4].values)
                max_gap_each_epoch = np.maximum.reduce(
                    [gap_column_2, gap_column_3, gap_column_4])  # get MaxGap

                # get Max Dice
                if max_dice > best_total_mean_ood and max_dice > valve_:
                    best_total_mean_ood = max_dice
                    best_total_mean_gap = max_gap_each_epoch[np.argmax(dice_values)]

                # Add FairMixup with FedAvg (Pretrain)
                if max_dice > best_model_dice and idx_ == 4:
                    best_model_dice = max_dice
                    best_model_path = global_file
        
        if best_total_mean_gap != -np.inf:
            methods_scores[weighted_labels[idx_]]['best_mean'].append(best_total_mean_gap)
            if idx_ == 4:
                step2_marker = True

    # FedAvg
    best_total_mean_base = -np.inf
    min_total_gap_base = np.inf
    min_gap_total_mean_base = 0
    best_total_mean_gap = -np.inf

    global_file = os.path.join(FedAvg_paths, 'result.csv')
    if os.path.exists(global_file):
        df_global = pd.read_csv(global_file, header=None)
        dice_values = df_global.iloc[:, 0].values
        max_dice = np.max(dice_values)  # get Max Dice

        # calculate MaxGap
        gap_column_2 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 2].values)
        gap_column_3 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 3].values)
        gap_column_4 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 4].values)
        max_gap_each_epoch = np.maximum.reduce(
            [gap_column_2, gap_column_3, gap_column_4])  # get MaxGap

        if max_dice > best_total_mean_base and max_dice > valve_:
            best_total_mean_base = max_dice
            best_total_mean_gap = max_gap_each_epoch[np.argmax(dice_values)]


    # Add FairMixup with FedAvg (Pretrain)
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

ax1.set_xlabel('Methods')
ax1.set_ylabel('MaxGap')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.10, 0.14)
plt.xticks(rotation=30)
plt.yticks(rotation=30)

plt.tight_layout()
plt.savefig('FundusFig2.pdf')
plt.show()
