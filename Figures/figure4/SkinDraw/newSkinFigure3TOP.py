import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from scipy.spatial import ConvexHull
from natsort import natsorted

labels = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova', 'FairMixup', 'FairFed', 'FlexFair']
seeds = range(5)  # all seeds

keys = ['ap']
key_dict = {'ap': 'Average Precision', 'acc': 'accuracy'}
dataset_label = ['Age DP', 'Age EO', 'Gender DP', 'Gender EO']  

pick_num = 19       # Get Top N
valve_value = 0.65  # filter low dice result

# set fairness target
datasets = ['Age_DP', 'Age_EO', 'Sex_DP', 'Sex_EO']  
ood_founder_dict = {'Age_DP': 'age-dp', 'Age_EO': 'age-eo', 'Sex_DP': 'sex-dp', 'Sex_EO': 'sex-eo'}
mixup_founder_dict = {'Age_DP': 'age-mixup', 'Age_EO': 'age-mixup', 'Sex_DP': 'sex-mixup', 'Sex_EO': 'sex-mixup'}

# # define Pareto Front
def identify_pareto(scores):
    pareto_front = []
    for i, (x1, y1) in enumerate(scores):
        if (x1, y1) == (0, 0):
            continue

        is_pareto = True
        for j, (x2, y2) in enumerate(scores):
            if i != j and x2 <= x1 and y2 >= y1:
                is_pareto = False
                break
        if is_pareto:
            pareto_front.append(i)
    return pareto_front


for key in keys:
    for idx_data, dataset in enumerate(datasets):
        print(dataset, key)
        if key == 'ap':
            paths_ = [
                f'your_data_path\\FedProx\\seed0',
                f'your_data_path\\Scaffold\\seed0',
                f'your_data_path\\FedNova\\seed0',
                f'your_data_path\\{mixup_founder_dict[dataset]}\\seed0',
                f'your_data_path\\{ood_founder_dict[dataset]}-FairFed\\seed0',
                f'your_data_path\\{ood_founder_dict[dataset]}-scaff-ap-unweighted\\seed0']

        # init dict for storage
        all_pareto_points = {}
        for label in labels:
            all_pareto_points[label] = []
        all_pareto_fronts = {}
        for label in labels:
            all_pareto_fronts[label] = []

        pick_num = 19

        # # init dict for store Dice and gap accrose all seeds
        top_dice_across_seeds = [[] for _ in range(pick_num)]
        top_gaps_across_seeds = [[] for _ in range(pick_num)]

        # across all seeds
        for seed in seeds:
            global_file = f'your_data_path\\FedAvg\\seed{seed}\\*\\result.csv'
            matched_files = glob.glob(global_file)

            if matched_files:
                df = pd.read_csv(matched_files[0])

                # Get MaxGap and Dice
                dice_values = df[key].values
                gaps = df[dataset].values

                # Get Top N Dice
                top_indices = np.argsort(dice_values)[-pick_num:][::-1]  
                top_dice_values = dice_values[top_indices]
                top_gap_values = gaps[top_indices]

                # storage
                for i in range(pick_num):
                    top_dice_across_seeds[i].append(top_dice_values[i])
                    top_gaps_across_seeds[i].append(top_gap_values[i])

        # Get Average, and dealing with finetune method
        for i in range(pick_num):
            avg_dice = np.mean(top_dice_across_seeds[i]) if top_dice_across_seeds[i] else None
            avg_gap = np.mean(top_gaps_across_seeds[i]) if top_gaps_across_seeds[i] else None
            print(avg_gap, avg_dice)
            if avg_dice > valve_value:
                all_pareto_points['FedAvg'].append((avg_gap, avg_dice))
                if i == 0:
                    all_pareto_points['FairMixup'].append((avg_gap, avg_dice))

        # Methods expect FedAvg
        for idx_, path_ in enumerate(paths_):
            # avoid beyond index
            if idx_ == 5:
                pick_num = 1
            elif idx_ == 3:
                pick_num = 2
            else:
                pick_num = 19
            for ood_path in glob.glob(path_):
                ood_df_paths = natsorted(glob.glob(os.path.join(ood_path, '*')))  # sort path

                for idx, base_df_path in ood_df_paths:
                    dice_values_across_seeds = []
                    gap_values_across_seeds = []

                    # # init dict for store Dice and gap accrose all seeds
                    top_dice_across_seeds = [[] for _ in range(pick_num)]
                    top_gaps_across_seeds = [[] for _ in range(pick_num)]

                    # Go across all seeds
                    for seed in seeds:
                        # get different seeds
                        seed_base_df_path = base_df_path.replace("seed0", f"seed{seed}")
                        seed_base_df_path = seed_base_df_path.replace("s0", f"s{seed}")

                        # fit path
                        if idx_ != 4:
                            global_file = os.path.join(seed_base_df_path, '*', 'result.csv')
                            matched_files = glob.glob(global_file)
                            df = pd.read_csv(matched_files[0])
                        else:
                            global_file = os.path.join(seed_base_df_path, 'result.csv')
                            df = pd.read_csv(global_file)

                        # Get MaxGap and Dice
                        dice_values = df[key].values
                        gaps = df[dataset].values

                        # Get Top N Dice
                        top_indices = np.argsort(dice_values)[-pick_num:][::-1]  
                        top_dice_values = dice_values[top_indices]
                        top_gap_values = gaps[top_indices]

                        # storage
                        for i in range(pick_num):
                            top_dice_across_seeds[i].append(top_dice_values[i])
                            top_gaps_across_seeds[i].append(top_gap_values[i])

                    # Get Average, and dealing with finetune method
                    for i in range(pick_num):
                        avg_dice = np.mean(top_dice_across_seeds[i]) if top_dice_across_seeds[i] else None
                        avg_gap = np.mean(top_gaps_across_seeds[i]) if top_gaps_across_seeds[i] else None
                        if avg_dice > valve_value:
                            all_pareto_points[labels[idx_ + 1]].append([avg_gap, avg_dice])
                            if i == 0 and idx_ == 1:
                                all_pareto_points['FlexFair'].append((avg_gap, avg_dice))

        # Calculate Pareto Front
        for label in labels:
            ood_pareto_points = np.array(all_pareto_points[label])
            pareto_indices = identify_pareto(ood_pareto_points)
            all_pareto_fronts[label] = ood_pareto_points[pareto_indices]

        # Plot
        plt.figure(figsize=(4, 4))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels) + 1))  

        
        for idx, label in enumerate(labels):
            sorted_indices = np.argsort(all_pareto_fronts[label][:, 0])  # sort with Gap
            sorted_pareto_front = all_pareto_fronts[label][sorted_indices]
            if dataset == 'Sex_DP' and label == 'FairFed':
                plt.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], '*', markersize=10, color=colors[idx + 1], label=label)
            else:
                plt.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], '-o', markerfacecolor='None', color=colors[idx + 1], label=label)

        print(all_pareto_points['FedAvg'])

        # Add info
        plt.xlabel(dataset_label[idx_data])
        plt.ylabel(key_dict[key])
        # plt.title('Baseline and OOD\'s Pareto Front')
        if dataset == 'Age_DP' or dataset == 'Sex_DP':
            plt.legend(loc='upper left')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('Skin_' + key + '_' + dataset + '-Fig3.pdf')
        plt.show()
