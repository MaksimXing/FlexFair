import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import seaborn as sns
from natsort import natsorted

datasets = ['Private', 'Public', 'Fundus']

dataset_names = ['Cervical Cancer', 'Polyp', ' Fundus Vascular']

labels = ['FlexFair', 'FedAvg', 'FedNova', 'FedProx', 'SCAFFOLD', 'FairMixup', 'FairFed']

methods_scores = {dataset: {label: [] for label in labels} for dataset in datasets}

valve = 20


for dataset in datasets:
    for seed in range(5):
        if dataset == 'Private':
            # fill in path here
            FedAvg_paths = f'FedAvg_paths\\seed{seed}'
            FedNova_paths = f'FedNova_paths\\seed{seed}\\'
            FedProx_paths = f'FedProx_paths\\seed{seed}\\'
            Scaffold_paths = f'Scaffold_paths\\seed{seed}\\'
            ood_df_paths = f'ood_df_paths\\seed{seed}\\'
            mixup_df_paths = f'mixup_df_paths\\seed{seed}\\'
            fairfed_df_paths = f'fairfed_df_paths\\seed{seed}\\'

            csv_name = 'global.csv'
        elif dataset == 'Public':

            FedAvg_paths = f'FedAvg_paths\\seed{seed}'
            FedNova_paths = f'FedNova_paths\\seed{seed}\\'
            FedProx_paths = f'FedProx_paths\\seed{seed}\\'
            Scaffold_paths = f'Scaffold_paths\\seed{seed}\\'
            ood_df_paths = f'ood_df_paths\\seed{seed}\\'
            mixup_df_paths = f'mixup_df_paths\\seed{seed}\\'
            fairfed_df_paths = f'fairfed_df_paths\\seed{seed}\\'

            csv_name = 'global.csv'
        elif dataset == 'Fundus':

            FedAvg_paths = f'FedAvg_paths\\seed{seed}'
            FedNova_paths = f'FedNova_paths\\seed{seed}\\'
            FedProx_paths = f'FedProx_paths\\seed{seed}\\'
            Scaffold_paths = f'Scaffold_paths\\seed{seed}\\'
            ood_df_paths = f'ood_df_paths\\seed{seed}\\'
            mixup_df_paths = f'mixup_df_paths\\seed{seed}\\'
            fairfed_df_paths = f'fairfed_df_paths\\seed{seed}\\'

            csv_name = 'result.csv'
        else:
            FedAvg_paths, FedNova_paths, FedProx_paths, Scaffold_paths, ood_df_paths, mixup_df_paths, fairfed_df_paths = [], [], [], [], [], [], []
            csv_name = ''

        weighted_paths = [FedNova_paths, FedProx_paths, Scaffold_paths, ood_df_paths, mixup_df_paths, fairfed_df_paths]
        weighted_labels = ["FedNova", "FedProx", "SCAFFOLD", "FlexFair", "FairMixup", "FairFed"]
        for idx_, path_ in enumerate(weighted_paths):
            ood_per_seed = []
            paths_ = natsorted(glob.glob(os.path.join(path_, '*')))  # 对路径进行排序
            for base_df_path in paths_:
                global_file = os.path.join(base_df_path, csv_name)
                print(base_df_path)
                if os.path.exists(global_file):
                    df_global = pd.read_csv(global_file, header=None)
                    dice_values = df_global.iloc[:, 0].values
                    ood_per_seed.extend(dice_values)
                else:
                    raise ArithmeticError
            # Get Top N
            if len(ood_per_seed) >= valve:
                top_3_dice = np.partition(ood_per_seed, len(ood_per_seed) - valve)[-valve:]
            else:
                # if length < N
                top_3_dice = ood_per_seed
            if type(top_3_dice[top_3_dice != 0]) == np.float64:
                methods_scores[dataset][weighted_labels[idx_]].extend([top_3_dice[top_3_dice != 0]])
            else:
                methods_scores[dataset][weighted_labels[idx_]].extend(top_3_dice[top_3_dice != 0])

        # for fedavg which doesn't has multiple para
        global_file = os.path.join(FedAvg_paths, csv_name)
        if os.path.exists(global_file):
            df_global = pd.read_csv(global_file)
            dice_values = df_global.iloc[:, 0].values
            dice_values = dice_values[dice_values != 0]
            if len(dice_values) >= valve:
                top_3_dice = np.partition(dice_values, len(dice_values) - valve)[-valve:]
            else:
                top_3_dice = dice_values
            methods_scores[dataset]['FedAvg'].extend(top_3_dice[top_3_dice != 0])
            # Considering FairMixup might based on FedAvg
            if dataset == 'Fundus':
                methods_scores[dataset]['FairMixup'].append(np.max(dice_values))
            elif dataset == 'Public':
                methods_scores[dataset]['FairMixup'].append(np.max(dice_values))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 7), sharey=False)

colors = sns.color_palette("Spectral", n_colors=len(labels))
count_ = 0
# 遍历每个数据集和对应的子图
for ax, dataset in zip(axes, datasets):
    data = [methods_scores[dataset][label] for label in labels]
    positions = 0.5 * np.arange(len(labels))

    if dataset == 'Private':
        ax.set_ylim(0.699, 0.821)
    if dataset == 'Public':
        ax.set_ylim(0.84, 0.901)
    if dataset == 'Fundus':
        ax.set_ylim(0.61, 0.67)

    for i, (pos, method_data) in enumerate(zip(positions, data)):
        color_idx = i % len(labels)

        parts = ax.violinplot(method_data, positions=[pos], showmeans=False, showmedians=False, showextrema=False)

        for pc in parts['bodies']:
            pc.set_facecolor(colors[color_idx])
            pc.set_edgecolor('gray')
            pc.set_alpha(0.3)
            pc.set_hatch('xxx')
        bp = ax.boxplot(method_data, positions=[pos], widths=0.2, patch_artist=True,
                        whis=1.5,
                        boxprops=dict(facecolor=colors[color_idx], color='black', alpha=0.7),
                        capprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        flierprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=2),
                        medianprops=dict(color='white', linewidth=2))

        mean_value = np.mean(method_data)
        ax.scatter(pos, mean_value, color='red', marker='x', zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_title(dataset_names[count_])
    count_ += 1

    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    sns.despine(ax=ax, offset=10, trim=True)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Patch(facecolor=colors[i], edgecolor='none', hatch='xxx', label=labels[i]) for i in
                   range(len(labels))]
legend_elements.append(Line2D([0], [0], color='red', marker='x', linestyle='None', label='Mean', markersize=8))
axes[-1].legend(handles=legend_elements, loc='lower left')
axes[0].set_ylabel('Dice Score')  # 替换为实际的 Y 轴标签名称

for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va('center')

plt.subplots_adjust(wspace=0.2, left=0.06, right=0.98, bottom=0.25)

plt.savefig('TOP10.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
