import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
from PIL import Image

# Step 1: Define your image directory
img_dir =
msk_dir =
sites = ['CHASEDB1', 'DRIVE', 'STARE']
mask_founder_name = ['1st_label', '1st_manual', '1st_labels_ah']
white_threshold = 220  # Define the threshold for white pixels

smoothed_overall_histogram_list = []


for idx, site in enumerate(sites):
    mask_dir = os.path.join(msk_dir, site, mask_founder_name[idx])
    # Step 2: Identify and read images and masks from the folder
    mask_files = [file for file in os.listdir(mask_dir) if
                  file.endswith(('.png', '.bmp', '.jpg', '.jpeg', '.ppm', '.tif', '.gif'))]

    # Initialize list to collect proportions
    overall_proportions = []

    for mask_file in mask_files:
        # Read the mask image
        mask_img = cv2.imread(os.path.join(mask_dir, mask_file), 0)  # Read in grayscale
        if mask_file.endswith('.gif'):
            gif_path = os.path.join(mask_dir, mask_file)
            gif = Image.open(gif_path)
            gif.seek(0)  # Go to the first frame
            frame = np.array(gif)
            mask_img = frame

        # Calculate total number of pixels in the mask image
        total_pixels = mask_img.shape[0] * mask_img.shape[1]

        # Threshold the mask image to identify white pixels
        _, binary_mask = cv2.threshold(mask_img, white_threshold - 1, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(binary_mask)

        # Calculate proportion of white pixels in the mask image
        proportion_white = white_pixels / total_pixels
        overall_proportions.append(proportion_white)

    # Step 3: Compute histogram over the proportions
    bins = np.linspace(0, 1, num=1000)
    overall_histogram, bin_edges = np.histogram(overall_proportions, bins=bins)
    # Normalize the histogram
    normalized_overall_histogram = overall_histogram / np.sum(overall_histogram)
    # Smooth the histogram
    smoothed_overall_histogram = gaussian_filter(normalized_overall_histogram, sigma=2)
    smoothed_overall_histogram_list.append(smoothed_overall_histogram)

# Save the data to a file
with open('Fundus_histogram_data.pkl', 'wb') as f:
    pickle.dump({'smoothed_histograms': smoothed_overall_histogram_list, 'bins': bins}, f)

print("Data saved successfully to 'Fundus_histogram_data.pkl'.")
