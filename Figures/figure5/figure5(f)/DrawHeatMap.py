import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from model import Model
import matplotlib
matplotlib.use('Agg')

def runDraw(args, base_path, pic_path, gt_path, image_name, model_path_1, model_path_2, output_pic_path, gt_attach, last_sentence=-4):

    info_1 = generate_segmentation_and_gradcam(args, model_path_1, os.path.join(base_path, pic_path, image_name))
    info_2 = generate_segmentation_and_gradcam(args, model_path_2, os.path.join(base_path, pic_path, image_name))

    # Load and process the ground truth mask
    gt_mask_path = os.path.join(base_path, gt_path, image_name[:last_sentence] + gt_attach)  # Adjust if mask has a different naming convention
    gt_image = Image.open(gt_mask_path).convert('L')  # Convert to grayscale
    # **Resize the mask to 352x352**
    gt_image = gt_image.resize((352, 352), Image.BILINEAR)  # Use appropriate resampling method

    gt_mask = np.array(gt_image)
    gt_mask = (gt_mask > 128).astype(np.uint8) * 255  # Binarize the mask


    Plot(info_1, info_2, gt_mask, output_pic_path)


def generate_segmentation_and_gradcam(args, model_path, image_path):
    # Load the model
    model = Model(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define a variable to store activations
    activations = None

    # Define a forward hook to capture activations
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        # Enable gradient computation for activations
        activations.retain_grad()

    # Register the hook on the target layer
    # Adjust the target layer as per your model architecture
    # Assuming the target layer is the last convolutional layer in self.fusion
    target_layer = model.fusion.linear4[0]  # Adjust this if needed
    handle = target_layer.register_forward_hook(forward_hook)

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((352, 352)),  # Match the training image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Adjust if different
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    input_img = Image.open(image_path).convert('RGB')
    tensor_img = preprocess(input_img).unsqueeze(0)  # Add batch dimension

    # Forward pass
    tensor_img = tensor_img
    tensor_img.requires_grad = True  # Enable gradients for input image
    output = model(tensor_img)
    # The model outputs a prediction map; adjust the size if needed
    output = F.interpolate(output, size=(352, 352), mode='bilinear', align_corners=True)

    # Generate segmentation mask
    segmentation = torch.sigmoid(output).detach().cpu().numpy()[0, 0]  # Convert to numpy array
    segmentation_mask = (segmentation > 0.5).astype(np.uint8) * 255  # Apply threshold

    # Compute the loss as the mean of the output
    pred_scalar = output.mean()

    # Backward pass
    model.zero_grad()
    pred_scalar.backward()

    # Get gradients and activations
    gradients = activations.grad  # [batch_size, channels, H, W]
    activations = activations.detach()  # [batch_size, channels, H, W]

    # Compute the weights
    weights = torch.mean(gradients, dim=(2, 3))  # [batch_size, channels]

    # Compute the Grad-CAM
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)  # [H, W]
    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]

    # Apply ReLU
    cam = torch.relu(cam)

    # Normalize the cam to [0,1]
    cam -= cam.min()
    cam /= cam.max()

    # Resize the cam to the size of the input image
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(352, 352), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Convert input image to numpy array
    img = input_img.resize((352, 352))
    img = np.array(img)

    # Convert the cam to RGB heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap onto the image
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.uint8(superimposed_img)

    # Release the hook
    handle.remove()

    return [img, segmentation_mask, superimposed_img]


def Plot(info_1, info_2, gt_mask, output_pic_path):

    # Plotting the results
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    info = [info_1, info_2]

    # Original Image
    axs[0].imshow(info_1[0])
    # axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Ground Truth Mask
    axs[1].imshow(info_1[0])  # Display the original image as the background
    axs[1].imshow(gt_mask, alpha=0.6, cmap='jet')  # Overlay the ground truth mask
    # axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    for index_ in range(0, 2):
        # Segmentation Result
        axs[index_ + 2].imshow(info[index_][0])
        axs[index_ + 2].imshow(info[index_][1], alpha=0.6, cmap='jet')
        # axs[index_ + 2].set_title('Segmentation Result')
        axs[index_ + 2].axis('off')

        # Grad-CAM Heatmap
        axs[index_ + 4].imshow(info[index_][2])
        # axs[index_ + 4].set_title('Grad-CAM Heatmap')
        axs[index_ + 4].axis('off')

    plt.tight_layout()
    plt.savefig(output_pic_path + '.png')
    # plt.show()
