import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import pickle

# List of .pkl files with corresponding site labels
data_files = [
    {'file': 'Public_histogram_data.pkl', 'sites': ['CVC-ClinicDB', 'Kvasir']},
    {'file': 'Private_histogram_data.pkl', 'sites': ['Center A', 'Center B', 'Center C', 'Center D']},
    {'file': 'Fundus_histogram_data.pkl', 'sites': ['CHASEDB1', 'DRIVE', 'STARE']}
]

Dataset_name = ['Polyp', 'Cervical Cancer', 'Fundus Vascular']

offset_increment= 1.1  # Adjusted offset to control the amount of overlap

fig, axes = plt.subplots(3, 1, figsize=(5, 6))

# First, collect the total number of histograms across all datasets
total_histograms = 0
for data_info in data_files:
    with open(data_info['file'], 'rb') as f:
        data = pickle.load(f)
    total_histograms += len(data['smoothed_histograms'])

# Generate colors from a rainbow colormap across all histograms
cmap = plt.get_cmap('rainbow', total_histograms)
# Generate a list of colors for all histograms
all_colors = [cmap(i / (total_histograms - 1)) for i in range(total_histograms)]

histogram_counter = 0  # To keep track of the histogram index for color assignment

for idx, data_info in enumerate(data_files):
    # Load the data from the saved file
    with open(data_info['file'], 'rb') as f:
        data = pickle.load(f)
    histograms = data['smoothed_histograms']
    bins = data['bins']
    labels = data_info['sites']

    # Normalize each histogram so that its maximum value is 1
    normalized_histogram_list = []
    for hist in histograms:
        max_value = np.max(hist)
        if max_value > 0:
            normalized_hist = hist / max_value
        else:
            normalized_hist = hist  # If the histogram is all zeros, leave it as is
        normalized_histogram_list.append(normalized_hist)

    # Total number of histograms in this data file
    n_samples = len(normalized_histogram_list)

    # Calculate y_offsets
    y_offsets = [i * offset_increment for i in range(n_samples)]

    # Calculate x_axis using bins
    x_axis = (bins[:-1] + bins[1:]) / 2  # Midpoints of bins

    ax = axes[idx]

    for i in range(n_samples):
        hist = normalized_histogram_list[i]
        y = hist + y_offsets[i]  # Add offset to the histogram

        # Get color from the all_colors list
        color = all_colors[histogram_counter]
        histogram_counter += 1

        # Plot the histogram and assign label for legend
        ax.fill_between(x_axis, y_offsets[i], y, color=color, alpha=0.4)
        ax.plot(x_axis, y, color=color, alpha=0.9, label=labels[i])

    ax.set_ylabel(Dataset_name[idx])
    # Adjust y-axis limit to accommodate the highest histogram plus offset
    total_height = y_offsets[-1] + np.max(normalized_histogram_list[-1]) + offset_increment
    ax.set_ylim(0, total_height)
    if idx != 2:
        ax.set_xscale('linear')
        ax.set_xlim(0, 0.06)
    else:
        ax.set_xscale('linear')
        ax.set_xlim(0, 0.15)

    # Remove y-axis ticks and labels for a cleaner appearance
    ax.set_yticks([])

    # Remove unnecessary spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    # Add legend to each subplot
    ax.legend(fontsize=8, loc='upper right')

axes[-1].set_xlabel('Proportion of Lesion Size in Image')
# Add shared y-axis label
# fig.supylabel('Normalized Frequency')  # Use fig.text(...) if using older Matplotlib

plt.tight_layout()
# Optionally save the figure
# plt.savefig('LesionSize.png')
plt.savefig('LesionSize.pdf')
plt.show()
