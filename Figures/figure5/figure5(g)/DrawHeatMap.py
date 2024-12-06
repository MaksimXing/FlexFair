# DrawHeatMap.py

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import method.resnet as resnet
import torch.nn as nn
import pandas as pd

def runDraw(args, base_path, gt_path, image_name, model_path_1, model_path_2, output_pic_path):

    tile_df = pd.read_csv(base_path + gt_path)

    matched_rows = tile_df[tile_df['isic_id'] == image_name[:-4]]

    if not matched_rows.empty:
        isic_id = matched_rows['isic_id'].iloc[0]
        benign_malignant = matched_rows['benign_malignant'].iloc[0]

    info_1 = generate_classification_and_gradcam(args, model_path_1, base_path + image_name)
    info_2 = generate_classification_and_gradcam(args, model_path_2, base_path + image_name)
    Plot(info_1, info_2, benign_malignant, output_pic_path)


def generate_classification_and_gradcam(args, model_path, image_path):
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Initialize the model
    model = resnet.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define a variable to store activations
    activations = None

    # Define a forward hook to capture activations from the target layer
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        # Enable gradient computation for activations
        activations.retain_grad()

    # Register the hook on the target layer
    # Typically, the last convolutional layer in ResNet is layer4
    target_layer = model.layer4  # Adjust if necessary
    handle = target_layer.register_forward_hook(forward_hook)

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as per your model's expectation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                             std=[0.229, 0.224, 0.225]),  # Standard ImageNet std
    ])

    # Load and preprocess the image
    input_image = Image.open(image_path).convert('RGB')
    tensor_image = preprocess(input_image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    output = model(tensor_image)
    pred_prob = torch.sigmoid(output).item()  # Assuming binary classification
    pred_label = 'Malignant' if pred_prob > 0.5 else 'Benign'  # Adjust labels as needed

    # Backward pass
    model.zero_grad()
    output.backward()

    # Get gradients from the target layer
    gradients = activations.grad  # [1, C, H, W]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # [C]

    # Weight the channels by corresponding gradients
    activations = activations.detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Compute the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)  # Normalize between 0-1

    # Convert heatmap to numpy
    heatmap = heatmap.numpy()

    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (input_image.width, input_image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap on the original image
    original_image = np.array(input_image)
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.uint8(superimposed_img)

    # Remove the hook
    handle.remove()

    return [original_image, superimposed_img, pred_label, pred_prob]

def Plot(info_1, info_2, benign_malignant, output_pic_path):
    # Plotting the results
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    info = [info_1, info_2]

    # Original Image
    axs[0].imshow(info_1[0])
    # axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(info_1[0])
    axs[1].text(50, 205, f'Ground Truth: \n{benign_malignant}',
                fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    # axs[1].set_title('Classification Result')
    axs[1].axis('off')

    for index_ in range(0, 2):
        # Classification Result
        axs[index_ + 2].imshow(info[index_][0])
        axs[index_ + 2].text(50, 205, f'Prediction: \n{info[index_][2]} ({info[index_][3]:.2f})',
                    fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        # axs[index_ + 2].set_title('Classification Result')
        axs[index_ + 2].axis('off')

        # Grad-CAM Heatmap
        axs[index_ + 4].imshow(info[index_][1])
        # axs[index_ + 4].set_title('Grad-CAM Heatmap')
        axs[index_ + 4].axis('off')

    plt.tight_layout()
    plt.savefig(output_pic_path + '.png')
