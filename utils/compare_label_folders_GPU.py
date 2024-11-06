import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

"""
Calculates scores like mIoU for 2 image folders: ground truth and predicted labels
calculates image wise and then calculates mean

Make sure to correctly define the class colours for actual and predicted 


"""

# Define colors for the actual labels
class_colors_actual = [
    [0, 149, 200],  # Label 0: sky
    [120, 187, 255],  # Label 1: obstacle
    [120, 113, 0],  # Label 2: vegetation/forest
    [228, 196, 80],  # Label 3: landscape_terrain/path
]

# Define colors for the predicted labels (example, replace with actual colors)
class_colors_predicted = [
        [135, 206, 235],  # Label 0: sky
        [128, 128, 128],  # Label 1: obstacles: rock, wall, vehicle, tree trunk, mountain, barn, building, roadside object, unlabelled
        [144, 238, 144],  # Label 2: vegetation
        [181, 101, 29],  # Label 3: landscape_terrain
        
    ]


def color_to_label(image, color_map):
    """Convert a color-labeled image to label indices based on the color map."""
    label_image = np.zeros(image.shape[:2], dtype=np.int32)
    for color, label in color_map.items():
        mask = (image == color).all(axis=-1)
        label_image[mask] = label
    return label_image

def extract_numeric_id(filename):
    """Extract the numeric ID from the filename, extracts just the numeric id from '1724236975_Camera0_class'."""
    return filename.split('_')[0]

def calculate_metrics_on_gpu(actual_path, predicted_path, color_map_actual, color_map_predicted, use_gpu=True):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Get filenames in each folder
    actual_files = {extract_numeric_id(f): f for f in os.listdir(actual_path)}
    predicted_files = {extract_numeric_id(f): f for f in os.listdir(predicted_path)}

    # Find common IDs in both folders
    common_ids = sorted(set(actual_files.keys()) & set(predicted_files.keys()))
    
    if not common_ids:
        print("No matching images found based on numerical IDs between the folders.")
        return None
    
    # Report unmatched files
    unmatched_actual = set(actual_files.keys()) - set(predicted_files.keys())
    unmatched_predicted = set(predicted_files.keys()) - set(actual_files.keys())
    if unmatched_actual:
        print(f"IDs missing in predicted folder: {unmatched_actual}")
    if unmatched_predicted:
        print(f"IDs missing in actual folder: {unmatched_predicted}")
    
    # Initialize list to store per-image IoU results
    results = []
    
    for image_id in tqdm(common_ids, desc="Processing image pairs"):
        # Load images based on matched numerical ID
        actual_image_path = os.path.join(actual_path, actual_files[image_id])
        predicted_image_path = os.path.join(predicted_path, predicted_files[image_id])
        
        actual_image_bgr = cv2.imread(actual_image_path)
        predicted_image_bgr = cv2.imread(predicted_image_path)

        # Convert images from BGR to RGB
        actual_image = cv2.cvtColor(actual_image_bgr, cv2.COLOR_BGR2RGB)
        predicted_image = cv2.cvtColor(predicted_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert color images to label images
        actual_labels = color_to_label(actual_image, color_map_actual)
        predicted_labels = color_to_label(predicted_image, color_map_predicted)

        # Convert to torch tensors and move to device
        actual_tensor = torch.tensor(actual_labels, dtype=torch.int64, device=device)
        predicted_tensor = torch.tensor(predicted_labels, dtype=torch.int64, device=device)
        
        # Calculate IoU for the entire image
        intersection = (actual_tensor == predicted_tensor) & (actual_tensor > 0)
        union = ((actual_tensor > 0) | (predicted_tensor > 0))
        iou = intersection.sum().float() / union.sum().float() if union.sum() > 0 else float('nan')

        # Calculate Dice score for the entire image
        actual_area = (actual_tensor > 0).sum().float()
        pred_area = (predicted_tensor > 0).sum().float()
        dice = (2 * intersection.sum().float() / (actual_area + pred_area)) if (actual_area + pred_area) > 0 else float('nan')

        # Append result for this file
        results.append({
            "image_id"  : image_id,
            "file_name" : actual_files[image_id],
            "image_iou" : iou.item(),
            "image_dice": dice.item()
        })

    # Save results to a DataFrame and write to CSV
    results_df = pd.DataFrame(results)
    output_dir = r"H:\semseg_fork\semantic-segmentation\logs\infer"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "segmentation_metrics_1.csv")
    results_df.to_csv(output_file, index=False)
    
    # Calculate and print overall mean IoU and Dice scores
    overall_mean_iou = results_df["image_iou"].mean()
    overall_mean_dice = results_df["image_dice"].mean()
    
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice Score: {overall_mean_dice:.4f}")
    
    return results_df


color_map_actual = {tuple(color): idx for idx, color in enumerate(class_colors_actual)}
color_map_predicted = {tuple(color): idx for idx, color in enumerate(class_colors_predicted)}


# Specify paths to folders with actual and predicted images
actual_folder = r"H:\semseg_fork\semantic-segmentation\data\compare_labels\labels_gt"
predicted_folder = r"H:\semseg_fork\semantic-segmentation\data\compare_labels\predicted"

# Calculate metrics with GPU support enabled
calculate_metrics_on_gpu(actual_folder, predicted_folder, color_map_actual, color_map_predicted, use_gpu=True)