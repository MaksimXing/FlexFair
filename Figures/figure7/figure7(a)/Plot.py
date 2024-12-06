import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
from matplotlib.colors import LinearSegmentedColormap

# List of .pkl files with corresponding site labels
data_files = [
    {'file': 'Public_histogram_data.pkl', 'sites': ['CVC-ClinicDB', 'Kvasir']},
    {'file': 'Private_histogram_data.pkl', 'sites': ['Center A', 'Center B', 'Center C', 'Center D']},
    {'file': 'Fundus_histogram_data.pkl', 'sites': ['CHASEDB1', 'DRIVE', 'STARE']}
]

Dataset_name = ['Polyp', 'Cervical Cancer', 'Fundus Vascular']

offset_increment = 1.1  # Adjusted offset to accommodate normalized histograms

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
fig, axes = plt.subplots(3, 1, figsize=(5, 6))

for idx, data_info in enumerate(data_files):
    # Load the data from the saved file
    with open(data_info['file'], 'rb') as f:
        data = pickle.load(f)
    histograms = data['smoothed_histograms']
    labels = data_info['sites']

    # Assuming that all histograms have the same x-axis (pixel intensities from 1 to 255)
    x_axis = np.arange(1, 256)

    # Normalize each histogram so that its maximum value is 1
    normalized_histogram_list = []
    for hist in histograms:
        max_value = np.max(hist)
        if max_value > 0:
            normalized_hist = hist / max_value
        else:
            normalized_hist = hist  # If the histogram is all zeros, leave it as is
        normalized_histogram_list.append(normalized_hist)

    n_samples = len(normalized_histogram_list)

    # Calculate y_offsets
    y_offsets = [i * offset_increment for i in range(n_samples)]

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

        # Removed the text labels next to each ridge

    # Set x-axis scale and limits
    if idx != 1:
        ax.set_xscale('linear')
        ax.set_xlim(1, 256)
    else:
        ax.set_xscale('linear')
        ax.set_xlim(1, 256)

    ax.set_ylabel(Dataset_name[idx])
    # Adjust y-axis limit to accommodate all histograms plus offsets
    total_height = y_offsets[-1] + np.max(normalized_histogram_list[-1]) + offset_increment
    ax.set_ylim(0, total_height)

    # Remove y-axis ticks and labels for a cleaner appearance
    ax.set_yticks([])

    # Remove unnecessary spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    # Hide x-axis labels for all but the last subplot
    if idx == len(axes) - 1:
        ax.set_xlabel('Pixel Intensity')

    # Add legend to each subplot
    ax.legend(fontsize=8, loc='upper right')
# Add shared y-axis label
# fig.supylabel('Normalized Frequency')  # Use fig.text(...) if using older Matplotlib

plt.tight_layout()
# Optionally save the figure
# plt.savefig('figure7(a).png')
plt.savefig('figure7(a).pdf')
plt.show()
