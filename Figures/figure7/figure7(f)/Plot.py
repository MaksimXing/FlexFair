import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

base_path = # put your dataset path here

# Load the two CSV files
bcn_data = pd.read_csv(os.path.join(base_path + 'BCN20000/BCN20000.csv'))
ham_data = pd.read_csv(os.path.join(base_path + 'HAM10000/HAM10000.csv'))

# Extract age data from both datasets
bcn_ages = bcn_data['age_approx'].dropna()
ham_ages = ham_data['age_approx'].dropna()

# Define age bins
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Group the data by age bins
bcn_age_distribution = pd.cut(bcn_ages, bins=age_bins).value_counts().sort_index()
ham_age_distribution = pd.cut(ham_ages, bins=age_bins).value_counts().sort_index()

# Create a plot
fig, ax = plt.subplots(figsize=(8, 3))

# Number of bins
N = len(age_bins) - 1

# Positions for the bars
ind = np.arange(N)

bar_width = 0.35  # Width of the bars

# Generate colors from a warm colormap
cmap = plt.cm.autumn  # Warm colors

# Generate colors for the bins
colors = [cmap(i / float(N - 1)) for i in range(N)]

# Create darker colors for the second dataset
darker_colors = []
for color in colors:
    # Convert to HSV to adjust brightness
    hsv = rgb_to_hsv(color[:3])
    hsv = list(hsv)
    hsv[2] = max(0, hsv[2] * 0.8)  # Decrease brightness by 30%
    darker_color = hsv_to_rgb(hsv)
    darker_colors.append(darker_color)

# Plot BCN20000 data
rects1 = ax.bar(ind - bar_width / 2, bcn_age_distribution.values, bar_width, color=colors, label='BCN20000')

# Plot HAM10000 data
rects2 = ax.bar(ind + bar_width / 2, ham_age_distribution.values, bar_width, color=darker_colors, label='HAM10000')

# Add labels, title, and legend
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
# ax.set_title('Age Distribution of Skin Cancer', fontsize=14)
ax.set_xticks(ind)
ax.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' for interval in bcn_age_distribution.index])
ax.legend()# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Adjust layout
plt.tight_layout()

# Display the plot
plt.savefig('figure7(f).pdf')
