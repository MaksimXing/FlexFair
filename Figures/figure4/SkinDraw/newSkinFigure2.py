import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from scipy.spatial import ConvexHull

labels = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova', 'FairMixup', 'FairFed', 'FlexFair']
seeds = range(5)  # go through all seeda

keys = ['acc']
key_dict = {'ap': 'Average Precision', 'acc': 'accuracy'}
dataset_label = ['Age DP', 'Age EO', 'Gender DP', 'Gender EO']  

pick_num = 19

gap_valve = 0.82  # Set Dice Valve

# set name
datasets = ['Age_DP', 'Age_EO', 'Sex_DP', 'Sex_EO']  
ood_founder_dict = {'Age_DP': 'age-dp', 'Age_EO': 'age-eo', 'Sex_DP': 'sex-dp', 'Sex_EO': 'sex-eo'}
mixup_founder_dict = {'Age_DP': 'age-mixup', 'Age_EO': 'age-mixup', 'Sex_DP': 'sex-mixup', 'Sex_EO': 'sex-mixup'}


# idetify Pareto Front
def identify_pareto(scores):
    pareto_front = []
    for i, (x1, y1, std1) in enumerate(scores):
        if (x1, y1) == (0, 0):
            continue

        is_pareto = True
        for j, (x2, y2, std2) in enumerate(scores):
            if i != j and x2 <= x1 and y2 >= y1:
                is_pareto = False
                break
        if is_pareto:
            pareto_front.append(i)
    return pareto_front


for key in keys:
    for idx_data, dataset in enumerate(datasets):
        if key == 'acc':
            paths_ = [
                f'your_path_on_result_Skin/FedProx/seed0',
                f'your_path_on_result_Skin/Scaffold/seed0',
                f'your_path_on_result_Skin/FedNova/seed0',
                f'your_path_on_result_Skin/{mixup_founder_dict[dataset]}/seed0',
                f'your_path_on_result_Skin/fairfed-{ood_founder_dict[dataset]}/seed0',
                f'your_path_on_result_Skin/{ood_founder_dict[dataset]}-phase1/seed0']

        # init
        all_pareto_points = {}
        for label in labels:
            all_pareto_points[label] = []
        all_pareto_fronts = {}
        for label in labels:
            all_pareto_fronts[label] = []

        # fedavg
        seed_dice_values = []
        seed_gap_values = []
        pick_num = 19

        # init array for storage
        top_dice_across_seeds = [[] for _ in range(pick_num)]
        top_gaps_across_seeds = [[] for _ in range(pick_num)]

        # go through all seeds
        for seed in seeds:
            global_file = f'your_path_on_result_Skin/FedAvg/seed{seed}/result.csv'
            matched_files = glob.glob(global_file)

            if matched_files:
                df = pd.read_csv(matched_files[0])

                # Get all dice and corresponding gap
                dice_values = df[key].values
                gaps = df[dataset].values

                # get top dice
                top_indices = np.argsort(dice_values)[-pick_num:][::-1]  # get top dice index
                top_dice_values = dice_values[top_indices]
                top_gap_values = gaps[top_indices]

                # add to array
                for i in range(pick_num):
                    top_dice_across_seeds[i].append(top_dice_values[i])
                    top_gaps_across_seeds[i].append(top_gap_values[i])

        # # calculate mean
        for i in range(pick_num):
            avg_dice = np.mean(top_dice_across_seeds[i]) if top_dice_across_seeds[i] else None
            avg_gap = np.mean(top_gaps_across_seeds[i]) if top_gaps_across_seeds[i] else None
            gap_std = np.std(top_gaps_across_seeds[i]) if top_gaps_across_seeds[i] else None
            all_pareto_points['FedAvg'].append((avg_gap, avg_dice, gap_std))
            if i == 0:
                all_pareto_points['FairMixup'].append((avg_gap, avg_dice, gap_std))

        # other baseline
        for idx_, path_ in enumerate(paths_): 
            if idx_ == 3:
                pick_num = 2
            else:
                pick_num = 19
            for ood_path in glob.glob(path_):
                if idx_ == 3 or idx_ == 4 or idx_ == 5:
                    from natsort import natsorted
                    ood_df_paths = natsorted(glob.glob(os.path.join(ood_path, '*')))  # sort path
                else:
                    ood_df_paths = sorted(glob.glob(os.path.join(ood_path, '*')))  # sort path

                for idx, base_df_path in ood_df_paths:
                    dice_values_across_seeds = []
                    gap_values_across_seeds = []

                    # init array for storage
                    top_dice_across_seeds = [[] for _ in range(pick_num)]
                    top_gaps_across_seeds = [[] for _ in range(pick_num)]

                    # go through all seeds
                    for seed in seeds:
                       # get seed path
                        seed_base_df_path = base_df_path.replace("seed0", f"seed{seed}")
                        seed_base_df_path = seed_base_df_path.replace("s0", f"s{seed}")

                        global_file = os.path.join(seed_base_df_path, 'result.csv')
                        df = pd.read_csv(global_file)
                        print(global_file)

                        # Get all dice and corresponding gap
                        dice_values = df[key].values
                        gaps = df[dataset].values

                        # get top dice
                        top_indices = np.argsort(dice_values)[-pick_num:][::-1]  # get top dice index
                        top_dice_values = dice_values[top_indices]
                        top_gap_values = gaps[top_indices]

                        # add to array
                        for i in range(pick_num):
                            top_dice_across_seeds[i].append(top_dice_values[i])
                            top_gaps_across_seeds[i].append(top_gap_values[i])

                    # # calculate mean
                    for i in range(pick_num):
                        avg_dice = np.mean(top_dice_across_seeds[i]) if top_dice_across_seeds[i] else None
                        avg_gap = np.mean(top_gaps_across_seeds[i]) if top_gaps_across_seeds[i] else None
                        gap_std = np.std(top_gaps_across_seeds[i]) if top_gaps_across_seeds[i] else None
                        all_pareto_points[labels[idx_ + 1]].append([avg_gap, avg_dice, gap_std])
                            

        # Calculate PF
        for label in labels:
            ood_pareto_points = np.array(all_pareto_points[label])
            pareto_indices = identify_pareto(ood_pareto_points)
            all_pareto_fronts[label] = ood_pareto_points[pareto_indices]

        gap_for_each_label = []
        gap_std_for_each_label = []

        for label in labels:
            # get PF in [gap, dice, gap_std]
            points = all_pareto_fronts[label]
            # # get Dice >= gap_valve
            valid_points = [p for p in points if p[1] >= gap_valve]
            if valid_points:
                # find min gap
                best_point = min(valid_points, key=lambda x: x[0])
                # read para
                gap_for_each_label.append(best_point[0])
                gap_std_for_each_label.append(best_point[2])
            else:
                # if NO dice >= gap_valve, return 0
                gap_for_each_label.append(0.0)
                gap_std_for_each_label.append(0.0)

        # plot
        fig, ax1 = plt.subplots(figsize=(4, 3))

        # set bar width
        width = 0.35
        x = np.arange(len(labels))  

        # set color
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels) + 1))  

        ax1.bar(x, gap_for_each_label, width,
                yerr=gap_std_for_each_label, capsize=5,
                color=colors[1:])

        
        # ax1.set_ylim(ylim_range[idx_data][0], ylim_range[idx_data][1])
        ax1.set_xlabel('Methods')
        ax1.set_ylabel(dataset_label[idx_data])
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)

        plt.xticks(rotation=30)
        plt.yticks(rotation=30)

        plt.tight_layout()
        plt.savefig('Skin_' + dataset + '-Fig2.pdf')
        plt.show()
