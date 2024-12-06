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

smoothed_overall_histogram_list = []

for idx, site in enumerate(sites):
    image_dir = os.path.join(img_dir, site, 'image')
    mask_dir = os.path.join(msk_dir, site, mask_founder_name[idx])
    # Step 2: Identify and read images and masks from the folder
    image_files = [file for file in os.listdir(image_dir) if file.endswith(('.png', '.bmp', '.jpg', '.jpeg', '.ppm', '.tif', '.gif'))]
    mask_files = [file for file in os.listdir(mask_dir) if file.endswith(('.png', '.bmp', '.jpg', '.jpeg', '.ppm', '.tif', '.gif'))]

    # Separate original and mask files assuming masks end with '_label'
    original_images = image_files
    mask_images = mask_files

    # Ensure the original and mask lists are matched up correctly
    matched_images = []
    for orig_file in original_images:
        base_name = os.path.splitext(orig_file)[0]
        if site == 'STARE':
            base_name = os.path.splitext(orig_file)[0][2:]
        if site == 'DRIVE':
            base_name = os.path.splitext(orig_file)[0][:2]
        matching_mask = next((m for m in mask_images if base_name in m), None)
        if matching_mask:
            matched_images.append((orig_file, matching_mask))

    # Step 3: Calculate and accumulate histograms for both areas and track non-zero pixels
    accumulated_masked_histogram = np.zeros((255, 1), dtype=np.float32)
    accumulated_unmasked_histogram = np.zeros((255, 1), dtype=np.float32)
    accumulated_overall_histogram = np.zeros((255, 1), dtype=np.float32)

    total_masked_pixels = 0
    total_unmasked_pixels = 0
    total_overall_pixels = 0

    for orig_file, mask_file in matched_images:
        # Read the original image and mask
        orig_img = cv2.imread(os.path.join(image_dir, orig_file), 0)  # Convert to grayscale
        mask_img = cv2.imread(os.path.join(mask_dir, mask_file), 0)  # Assuming mask is also grayscale
        if mask_file.endswith('.gif'):
            gif_path = os.path.join(mask_dir, mask_file)
            gif = Image.open(gif_path)
            gif.seek(0)  # Go to the first frame
            frame = np.array(gif)
            mask_img = frame

        # Apply the mask to the original image for masked and unmasked images
        masked_img = cv2.bitwise_and(orig_img, orig_img, mask=mask_img)
        unmasked_img = cv2.bitwise_and(orig_img, orig_img, mask=cv2.bitwise_not(mask_img))

        # Calculate histogram for masked and unmasked areas, excluding the zero intensity
        masked_histogram = cv2.calcHist([masked_img], [0], None, [255], [1, 256])
        unmasked_histogram = cv2.calcHist([unmasked_img], [0], None, [255], [1, 256])
        overall_histogram = cv2.calcHist([orig_img], [0], None, [255], [1, 256])

        # Accumulate histograms
        accumulated_masked_histogram += masked_histogram
        accumulated_unmasked_histogram += unmasked_histogram
        accumulated_overall_histogram += overall_histogram

        # Count non-zero pixels in masked and unmasked areas
        total_masked_pixels += cv2.countNonZero(masked_img)
        total_unmasked_pixels += cv2.countNonZero(unmasked_img)
        total_overall_pixels += cv2.countNonZero(orig_img)

    # Step 4: Normalize and smooth the accumulated histograms
    normalized_masked_histogram = accumulated_masked_histogram / total_masked_pixels
    normalized_unmasked_histogram = accumulated_unmasked_histogram / total_unmasked_pixels
    normalized_overall_histogram = accumulated_overall_histogram / total_overall_pixels

    smoothed_masked_histogram = gaussian_filter(normalized_masked_histogram.flatten(), sigma=2)
    smoothed_unmasked_histogram = gaussian_filter(normalized_unmasked_histogram.flatten(), sigma=2)
    smoothed_overall_histogram = gaussian_filter(normalized_overall_histogram.flatten(), sigma=2)

    smoothed_overall_histogram_list.append(smoothed_overall_histogram)

# Save the data to a file
with open('Fundus_histogram_data.pkl', 'wb') as f:
    pickle.dump({'smoothed_histograms': smoothed_overall_histogram_list}, f)

print("Data saved successfully to 'Fundus_histogram_data.pkl'.")
