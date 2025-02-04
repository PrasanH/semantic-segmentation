import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Define colors for the ground truth (actual) and predicted labels
class_colors_actual = [
    [0, 149, 200],  # Label 0: sky
    [120, 187, 255],  # Label 1: obstacle
    [120, 113, 0],  # Label 2: vegetation/forest
    [228, 196, 80],  # Label 3: landscape_terrain/path
]

class_colors_predicted = [
        [135, 206, 235],  # Label 0: sky
        [128, 128, 128],  # Label 1: obstacles: rock, wall, vehicle, tree trunk, mountain, barn, building, roadside object, unlabelled
        [144, 238, 144],  # Label 2: vegetation
        [181, 101, 29],  # Label 3: landscape_terrain
        
    ]

# Convert color arrays to dictionaries for easier mapping
def create_color_to_label_map(class_colors):
    return {tuple(color): idx for idx, color in enumerate(class_colors)}

color_map_actual = create_color_to_label_map(class_colors_actual)
color_map_predicted = create_color_to_label_map(class_colors_predicted)

def color_to_label(image, color_map):
    """Convert a color-labeled image to label indices based on the color map."""
    label_image = np.zeros(image.shape[:2], dtype=np.int32)
    for color, label in color_map.items():
        mask = (image == color).all(axis=-1)
        label_image[mask] = label
    return label_image

def calculate_miou(gt_labels, pred_labels):
    """Calculate mIoU between ground truth and predicted labels."""
    gt_tensor = torch.tensor(gt_labels, dtype=torch.int64)
    pred_tensor = torch.tensor(pred_labels, dtype=torch.int64)

    intersection = (gt_tensor == pred_tensor) & (gt_tensor > 0)
    union = ((gt_tensor > 0) | (pred_tensor > 0))
    miou = intersection.sum().float() / union.sum().float()
    return miou.item()


ground_truth_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/test/labels/"
predicted_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/out/1_res101_nov4_e10_526_default/test/"

# Example file names (replace with actual file names or loop through files)
gt_filename = '1721394021_Camera0_class.png'
pred_filename = '1721394021_Camera0_visible.png'



# Read the ground truth and predicted images
gt_image_path = os.path.join(ground_truth_folder, gt_filename)
pred_image_path = os.path.join(predicted_folder, pred_filename)

gt_image = cv2.imread(gt_image_path)
pred_image = cv2.imread(pred_image_path)

# Convert the images from BGR to RGB (as OpenCV loads images in BGR by default)
gt_image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

# Check if the images were loaded successfully
if gt_image_rgb is None:
    print(f"Error: Could not load ground truth image: {gt_image_path}")
elif pred_image_rgb is None:
    print(f"Error: Could not load predicted image: {pred_image_path}")
else:
    # Convert color images to label indices
    gt_labels = color_to_label(gt_image_rgb, color_map_actual)
    pred_labels = color_to_label(pred_image_rgb, color_map_predicted)
    
    # Calculate mIoU
    miou = calculate_miou(gt_labels, pred_labels)
    
    # Plot both images side by side with mIoU
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(gt_image_rgb)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(pred_image_rgb)
    axes[1].set_title("Prediction")
    axes[1].axis('off')
    
    plt.suptitle(f"mIoU: {miou:.4f}", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"Mean IoU: {miou:.4f}")