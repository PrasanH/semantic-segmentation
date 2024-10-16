import os
import cv2  # or use PIL if preferred
import numpy as np
from sklearn.metrics import jaccard_score  # For IoU
import re



def extract_initial_number(filename):
    return filename.split('_')[0]
'''
def compute_iou(gt_mask, pred_mask, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        intersection = np.logical_and(gt_mask == cls, pred_mask == cls).sum()
        union = np.logical_or(gt_mask == cls, pred_mask == cls).sum()
        print(f"Class {cls}: Intersection: {intersection}, Union: {union}")
        
        if union == 0:
            iou_per_class.append(0.0)  # No overlap
        else:
            iou = intersection / union
            iou_per_class.append(iou)
    
    mean_iou = np.mean(iou_per_class)
    print(f"Mean IoU: {mean_iou}")  # Debugging info
    return mean_iou  # Return mean IoU over all classes
'''
def compute_iou(gt_mask, pred_mask):
    """
    Compute the mean Intersection over Union (IoU) for the classes present in gt_mask.

    Parameters:
    - gt_mask: Ground truth mask (2D array).
    - pred_mask: Predicted mask (2D array).

    Returns:
    - mean_iou: Mean IoU across classes present in gt_mask.
    """
    # Get unique classes from the ground truth mask
    unique_classes = np.unique(gt_mask)
    iou_per_class = []

    for cls in unique_classes:
        # Calculate intersection and union
        intersection = np.logical_and(gt_mask == cls, pred_mask == cls).sum()
        union = np.logical_or(gt_mask == cls, pred_mask == cls).sum()

        # Debugging output for intersection and union
        print(f"Class {cls}: Intersection: {intersection}, Union: {union}")

        # Calculate IoU for the current class
        if union == 0:
            iou_per_class.append(0.0)  # No overlap
        else:
            iou = intersection / union
            iou_per_class.append(iou)

    # Calculate mean IoU, handling the case of no unique classes
    if iou_per_class:  # ensure there's at least one IoU calculated
        mean_iou = np.mean(iou_per_class)
    else:
        mean_iou = 0.0  # default to 0 if no classes were considered

    # Print the mean IoU for debugging purposes
    print(f"Mean IoU: {mean_iou}")
    return mean_iou  # Return mean IoU over all classes




def compute_dice(gt_mask, pred_mask, num_classes):
    dice_per_class = []
    for cls in range(num_classes):
        intersection = np.logical_and(gt_mask == cls, pred_mask == cls).sum()
        gt_count = (gt_mask == cls).sum()
        pred_count = (pred_mask == cls).sum()
        print(f"Class {cls}: Intersection: {intersection}, GT Count: {gt_count}, Pred Count: {pred_count}")  # Debugging info

        if gt_count + pred_count == 0:
            dice_per_class.append(0.0)  # If both are empty, consider Dice as 0
        else:
            dice = (2. * intersection) / (gt_count + pred_count)
            dice_per_class.append(dice)

    mean_dice = np.mean(dice_per_class)
    print(f"Mean Dice: {mean_dice}")  # Debugging info
    return mean_dice  # Return mean Dice coefficient over all classes

# Function to convert RGB masks to class indices (assume each RGB color corresponds to a class)
def rgb_to_class(mask_rgb, class_colors):
    mask_class = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    for idx, color in enumerate(class_colors):
        mask_class[np.all(mask_rgb == color, axis=-1)] = idx
    return mask_class


ground_truth_folder = r"C:\Users\prasa\Downloads\rrlab_download\simulated\gt_labels_1"
predicted_folder = r'C:\Users\prasa\Downloads\rrlab_download\simulated\prediction_1'


class_colors = [
        [0, 149, 200],  # Label 0: sky
        [120, 187, 255],  # Label 1: obstacle
        [120, 113, 0],  # Label 2: vegetation/forest
        [228, 196, 80],  # Label 3: landscape_terrain/path
    ]


num_classes = len(class_colors)

# Get filenames from both folders
gt_files = sorted(os.listdir(ground_truth_folder))
pred_files = sorted(os.listdir(predicted_folder))


# Create dictionaries with the initial number as the key
gt_dict = {extract_initial_number(f): f for f in gt_files if 'class' in f}
pred_dict = {extract_initial_number(f): f for f in pred_files if 'visible' in f}

# Ensure both folders contain the same initial numbers
common_keys = set(gt_dict.keys()).intersection(set(pred_dict.keys()))

iou_scores = []
dice_scores = []

# Loop through the common keys and compare masks
for key in common_keys:
    gt_mask_name = gt_dict[key]
    pred_mask_name = pred_dict[key]

    gt_mask_path = os.path.join(ground_truth_folder, gt_mask_name)
    pred_mask_path = os.path.join(predicted_folder, pred_mask_name)

    # Read RGB masks
    gt_mask_bgr = cv2.imread(gt_mask_path)
    gt_mask_rgb = cv2.cvtColor(gt_mask_bgr, cv2.COLOR_BGR2RGB)
    pred_mask_bgr = cv2.imread(pred_mask_path)
    pred_mask_rgb = cv2.cvtColor(pred_mask_bgr, cv2.COLOR_BGR2RGB)

    # Check if the images loaded correctly
    if gt_mask_rgb is None:
        print(f"Error: Could not load ground truth mask: {gt_mask_name}")
        continue
    if pred_mask_rgb is None:
        print(f"Error: Could not load predicted mask: {pred_mask_name}")
        continue

    # Ensure the masks have the same shape
    if gt_mask_rgb.shape != pred_mask_rgb.shape:
        print(f"Shape mismatch: {gt_mask_name} and {pred_mask_name}")
        continue

    # Convert RGB masks to class indices
    gt_mask_class = rgb_to_class(gt_mask_rgb, class_colors)
    pred_mask_class = rgb_to_class(pred_mask_rgb, class_colors)

    # Compute IoU and Dice coefficient for multi-class segmentation
    iou = compute_iou(gt_mask_class, pred_mask_class)
    dice = compute_dice(gt_mask_class, pred_mask_class, num_classes)

    # Append results
    iou_scores.append(iou)
    dice_scores.append(dice)

    print(f"Processed {gt_mask_name} | IoU: {iou:.4f}, Dice: {dice:.4f}")

# Summary statistics
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)

print(f"\nMean IoU: {mean_iou:.4f}")
print(f"Mean Dice Coefficient: {mean_dice:.4f}")

