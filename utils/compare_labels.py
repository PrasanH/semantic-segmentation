import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

"""
Calculates scores like mIoU for 2 image folders: ground truth and predicted labels

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


# Convert color arrays to dictionaries for easier mapping
def create_color_to_label_map(class_colors):
    return {tuple(color): idx for idx, color in enumerate(class_colors)}

color_map_actual = create_color_to_label_map(class_colors_actual)
color_map_predicted = create_color_to_label_map(class_colors_predicted)

#print("color map actual", color_map_actual)
#print("color map predicted", color_map_predicted)



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

def calculate_metrics_on_gpu(actual_path, predicted_path, use_gpu=True):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Get filenames in each folder
    actual_files = {extract_numeric_id(f): f for f in os.listdir(actual_path)}
    predicted_files = {extract_numeric_id(f): f for f in os.listdir(predicted_path)}
    
    #print("actual files", actual_files)
    #print("predicted files", predicted_files)

    # Find common IDs in both folders
    common_ids = sorted(set(actual_files.keys()) & set(predicted_files.keys()))
    #print("common_ids", common_ids)
    
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
    
    # Initialize lists to store results
    results = []
    
    for image_id in common_ids:
        # Load images based on matched numerical ID
        actual_image_path = os.path.join(actual_path, actual_files[image_id])
        predicted_image_path = os.path.join(predicted_path, predicted_files[image_id])
        
        actual_image = cv2.imread(actual_image_path)
        predicted_image = cv2.imread(predicted_image_path)
        
        # Convert color images to label images
        actual_labels = color_to_label(actual_image, color_map_actual)
        predicted_labels = color_to_label(predicted_image, color_map_predicted)

        # Convert to torch tensors and move to device
        actual_tensor = torch.tensor(actual_labels, dtype=torch.int64, device=device)
        predicted_tensor = torch.tensor(predicted_labels, dtype=torch.int64, device=device)
        
        # Calculate mIoU and Dice score per class
        per_class_iou = []
        per_class_dice = []
        
        for label in range(len(class_colors_actual)):
            intersection = ((actual_tensor == label) & (predicted_tensor == label)).sum().float()
            union = ((actual_tensor == label) | (predicted_tensor == label)).sum().float()
            actual_area = (actual_tensor == label).sum().float()
            pred_area = (predicted_tensor == label).sum().float()
            
            iou = (intersection / union).item() if union > 0 else float('nan')
            dice = (2 * intersection / (actual_area + pred_area)).item() if (actual_area + pred_area) > 0 else float('nan')
            
            per_class_iou.append(iou)
            per_class_dice.append(dice)
        
        # Average IoU and Dice score for the image
        avg_iou = np.nanmean(per_class_iou)
        avg_dice = np.nanmean(per_class_dice)
        
        # Append result for this file
        results.append({
            "image_id": image_id,
            "file_name": actual_files[image_id],
            "mean_iou": avg_iou,
            "mean_dice": avg_dice,
            **{f"class_{label}_iou": iou for label, iou in enumerate(per_class_iou)},
            **{f"class_{label}_dice": dice for label, dice in enumerate(per_class_dice)}
        })

    # Save results to a DataFrame and write to CSV
    results_df = pd.DataFrame(results)
    output_dir = "/home/pdhegde/semseg_git_fork/semantic-segmentation/logs/infer"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "segmentation_metrics.csv")
    results_df.to_csv(output_file, index=False)
    
    # Calculate and print overall mean IoU and Dice scores
    overall_mean_iou = results_df["mean_iou"].mean()
    overall_mean_dice = results_df["mean_dice"].mean()
    
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice Score: {overall_mean_dice:.4f}")
    
    return results_df

# Specify paths to folders with actual and predicted images
actual_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/test/labels/"
predicted_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/out/1_res101_nov4_e10_526_default/test/"

# Calculate metrics with GPU support enabled
calculate_metrics_on_gpu(actual_folder, predicted_folder, use_gpu=True)