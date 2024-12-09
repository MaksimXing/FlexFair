import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from scipy.spatial import ConvexHull
import matplotlib
from natsort import natsorted

matplotlib.use('TkAgg')

# define method label
labels = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova', 'FairMixup', 'FairFed', 'FlexFair']
seeds = range(5)  # all seeds

pick_num = 10   # Get Top N

valve = 0.635 # filter low dice result

paths_ = [
    f'E:your_data_path\\FedProxmiu\\seed0',
    f'E:your_data_path\\Scaffoldmiu\\seed0',
    f'E:your_data_path\\FedNovamiu\\seed0',
    f'E:your_data_path\\FairMixup_step2\\seed0',
    f'E:your_data_path\\FairFed\\seed0',
    f'E:your_data_path\\FedAvgOOD_step2\\seed0']

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


# init dict for storage
baseline_max_points = {}
baseline_max_points['FedAvg'] = {"avg_top_points": []}
all_pareto_points = {}
for label in labels:
    all_pareto_points[label] = []
all_pareto_fronts = {}
for label in labels:
    all_pareto_fronts[label] = []

# # init dict for store Dice and gap accrose all seeds
top_dice_across_seeds = [[] for _ in range(pick_num)]
top_gaps_across_seeds = [[] for _ in range(pick_num)]

for seed in seeds:
    pick_num = 10
    global_file = f'E:your_data_path\\FedAvg\\seed{seed}\\result.csv'

    if os.path.exists(global_file):
        df = pd.read_csv(global_file, header=None)

        # Calculate Max Gap
        gap_column_2 = np.abs(df.iloc[:, 0].values - df.iloc[:, 2].values)  # Client 1
        gap_column_3 = np.abs(df.iloc[:, 0].values - df.iloc[:, 3].values)  # Client 2
        gap_column_4 = np.abs(df.iloc[:, 0].values - df.iloc[:, 4].values)
        max_gap_each_row = np.maximum.reduce(
            [gap_column_2, gap_column_3, gap_column_4])  # get MaxGap

        # Get MaxGap and Dice
        dice_values = df.iloc[:, 0].values
        gaps = max_gap_each_row

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
    if avg_dice > valve:
        all_pareto_points['FedAvg'].append((avg_gap, avg_dice))
        if i == 0:
            all_pareto_points['FairMixup'].append((avg_gap, avg_dice))
            all_pareto_points['FlexFair'].append((avg_gap, avg_dice))

# Methods expect FedAvg
for idx_, path_ in enumerate(paths_):
    if idx_ == 3 or idx_ == 5:
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

            for seed in seeds:
                # get different seeds
                seed_base_df_path = base_df_path.replace("seed0", f"seed{seed}")
                seed_base_df_path = seed_base_df_path.replace("s0", f"s{seed}")

                global_file = os.path.join(seed_base_df_path, 'result.csv')

                if os.path.exists(global_file):
                    print(global_file)
                    df = pd.read_csv(global_file, header=None)

                    # Calculate MaxGap
                    gap_column_2 = np.abs(df.iloc[:, 0].values - df.iloc[:, 2].values)  # Client 1
                    gap_column_3 = np.abs(df.iloc[:, 0].values - df.iloc[:, 3].values)  # Client 2
                    gap_column_4 = np.abs(df.iloc[:, 0].values - df.iloc[:, 4].values)
                    max_gap_each_row = np.maximum.reduce(
                        [gap_column_2, gap_column_3, gap_column_4])  # get MaxGap

                    # Get MaxGap and Dice
                    dice_values = df.iloc[:, 0].values
                    gaps = max_gap_each_row

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
                if avg_dice > valve:
                    all_pareto_points[labels[idx_ + 1]].append([avg_gap, avg_dice])

for label in labels:
    ood_pareto_points = np.array(all_pareto_points[label])
    pareto_indices = identify_pareto(ood_pareto_points)
    all_pareto_fronts[label] = ood_pareto_points[pareto_indices]

# Plot
plt.figure(figsize=(4, 3))
colors = plt.cm.rainbow(np.linspace(0, 1, len(labels) + 1))  


for idx, label in enumerate(labels):
    sorted_indices = np.argsort(all_pareto_fronts[label][:, 0])  # sort with Gap
    sorted_pareto_front = all_pareto_fronts[label][sorted_indices]
    plt.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], '-o', markerfacecolor='None', color=colors[idx + 1], label=label)


# Add info
plt.xlabel('Max Gap')
plt.ylabel('Dice')
# plt.title('Baseline and OOD\'s Pareto Front')
plt.legend(loc='lower right', prop = {'size':8})
plt.yticks(rotation=45)
plt.xlim(0.105, 0.135)
# plt.ylim(0.64, 0.665)
plt.tick_params(axis='x', labelsize=8)
plt.tight_layout()
plt.grid(True)
plt.savefig('FundusFig3.pdf')
plt.show()
