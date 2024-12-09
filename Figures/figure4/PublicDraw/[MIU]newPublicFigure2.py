import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from natsort import natsorted

valve_ = 0.881 # filter low dice result


labels = ['FedAvg', 'FedProx','FedNova', 'FairMixup', 'FairFed', 'FlexFair']
methods_scores = {label: {'best_mean': [], 'min_mean': []} for label in labels}


for seed in range(5):

    step2_marker = False
    
    FedAvg_paths = f'your_data_path\\FedAvg\\seed{seed}'
    FedNova_paths = f'your_data_path\\FedNova-miu\\seed{seed}\\'
    FedProx_paths = f'your_data_path\\FedProx-miu\\seed{seed}\\'
    Scaffold_paths = f'your_data_path\\Scaffold-miu\\seed{seed}\\'

    ood_df_paths = f'your_data_path\\FedAvg_OOD\\seed{seed}\\'

    mixup_df_paths = f'your_data_path\\FairMixup_step2\\seed{seed}\\'
    fairfed_df_paths = f'your_data_path\\fairfed\\seed{seed}\\'


    
    weighted_paths = [FedNova_paths, FedProx_paths, ood_df_paths, fairfed_df_paths, mixup_df_paths]
    weighted_labels = ["FedNova", "FedProx", "FlexFair", 'FairFed', "FairMixup"]
    for idx_, path_ in enumerate(weighted_paths):
        best_total_mean_ood = -np.inf
        min_total_gap_ood = np.inf
        min_gap_total_mean_ood = 0
        best_total_mean_gap = -np.inf

        paths = natsorted(glob.glob(os.path.join(path_, '*')))  # sort path

        for idx, base_df_path in paths:
            global_file = os.path.join(base_df_path, 'global.csv')
            if os.path.exists(global_file):
                df_global = pd.read_csv(global_file)
                dice_values = df_global.iloc[:, 0].values
                max_dice = np.max(dice_values)  # get Max Dice

                # calculate MaxGap
                gap_column_2 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 1].values)
                gap_column_3 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 2].values)
                max_gap_each_epoch = np.maximum(gap_column_2, gap_column_3)

                # get Max Dice
                if max_dice > best_total_mean_ood and max_dice > valve_:
                    best_total_mean_ood = max_dice
                    best_total_mean_gap = max_gap_each_epoch[np.argmax(dice_values)]

        if best_total_mean_gap != -np.inf:
            methods_scores[weighted_labels[idx_]]['best_mean'].append(best_total_mean_gap)
            if idx_ == 4:
                step2_marker = True

    # Deal with FedAvg
    best_total_mean_base = -np.inf
    min_total_gap_base = np.inf
    min_gap_total_mean_base = 0
    best_total_mean_gap = -np.inf

    global_file = os.path.join(FedAvg_paths, 'global.csv')
    if os.path.exists(global_file):
        df_global = pd.read_csv(global_file)
        dice_values = df_global.iloc[:, 0].values
        max_dice = np.max(dice_values)  # get Max Dice

        # calculate MaxGap
        gap_column_2 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 1].values)
        gap_column_3 = abs(df_global.iloc[:, 0].values - df_global.iloc[:, 2].values)
        max_gap_each_epoch = np.maximum(gap_column_2, gap_column_3)

        # get Max Dice
        if max_dice > best_total_mean_base and max_dice > valve_:
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
colors = plt.cm.rainbow(np.linspace(0, 1, 7 + 1))  
idx = [1, 2,4, 5,6,7]


ax1.bar(x, [np.mean(methods_scores[label]['best_mean']) for label in labels], width,
        yerr=[np.std(methods_scores[label]['best_mean']) for label in labels], capsize=5,
        color=colors[idx])

# Add Info
ax1.set_xlabel('Methods')
ax1.set_ylabel('MaxGap')
# ax1.set_title('MaxGap for Min MaxGap')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
# ax1.legend(loc='upper right')
ax1.set_ylim(0.0, 0.022)
plt.xticks(rotation=30)
plt.yticks(rotation=30)

plt.tight_layout()
plt.savefig('PublicFig2.pdf')
plt.show()
