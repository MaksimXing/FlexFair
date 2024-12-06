import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle

# Step 1: Define your image directory
img_dir =
msk_dir =
sites = ['site_CVC', 'site_KVA']
white_threshold = 220  # Define the threshold for white pixels

smoothed_overall_histogram_list = []

for site in sites:
    image_dir = os.path.join(img_dir, site)
    mask_dir = os.path.join(msk_dir, site)
    mask_images = [file for file in os.listdir(mask_dir) if file.endswith(('.png', '.bmp', '.jpg', '.jpeg'))]

    # Initialize list to collect proportions
    overall_proportions = []

    for mask_file in mask_images:
        # Read the mask image
        mask_img = cv2.imread(os.path.join(image_dir, mask_file), 0)  # Read in grayscale

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
with open('Public_histogram_data.pkl', 'wb') as f:
    pickle.dump({'smoothed_histograms': smoothed_overall_histogram_list, 'bins': bins}, f)

print("Data saved successfully to 'Public_histogram_data.pkl'.")
