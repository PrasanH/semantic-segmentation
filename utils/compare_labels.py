import os
import cv2  # or use PIL if preferred
import numpy as np
from sklearn.metrics import jaccard_score  # For IoU
import re


def extract_initial_number(filename):
    """
    Helper function to extract the initial number from a filename
    """
    return re.match(r'^\d+', filename).group()


def compute_iou(gt_mask, pred_mask):
    """ Computes Intersection over Union (IoU) """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def compute_dice(gt_mask, pred_mask):
    """ Computes Dice Coefficient """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    dice = (2. * intersection) / (gt_mask.sum() + pred_mask.sum()) if (gt_mask.sum() + pred_mask.sum()) != 0 else 0
    return dice

# Paths to the ground truth and predicted mask folders
ground_truth_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/simulated/labels/'
predicted_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/out/oct2_1resnet101/simulated/'

# Ensure both folders have the same number of images and order
gt_masks = sorted(os.listdir(ground_truth_folder))
pred_masks = sorted(os.listdir(predicted_folder))

gt_numbers = {extract_initial_number(gt): gt for gt in gt_masks}
pred_numbers = {extract_initial_number(pred): pred for pred in pred_masks}

# Make sure both folders have the same number of masks
assert len(gt_numbers) == len(pred_numbers), "Mismatch in the number of masks between the two folders!"

# Initialize metrics
iou_scores = []
dice_scores = []

for number in gt_numbers:
    # Load ground truth and predicted masks
    gt_mask_name = gt_numbers[number]
    pred_mask_name = pred_numbers[number]
    
    gt_mask_path = os.path.join(ground_truth_folder, gt_mask_name)
    pred_mask_path = os.path.join(predicted_folder, pred_mask_name)

    # Read images as binary masks (0 or 1 values)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the masks have the same shape
    assert gt_mask.shape == pred_mask.shape, f"Shape mismatch: {gt_mask_name} and {pred_mask_name}"

    # Convert masks to binary values (0 or 1)
    gt_mask_binary = gt_mask // 255  # Convert 255 values to 1s
    pred_mask_binary = pred_mask // 255

    # Compute IoU and Dice coefficient
    iou = compute_iou(gt_mask_binary, pred_mask_binary)
    dice = compute_dice(gt_mask_binary, pred_mask_binary)

    # Append results
    iou_scores.append(iou)
    dice_scores.append(dice)

    print(f"Processed {gt_mask_name} | IoU: {iou:.4f}, Dice: {dice:.4f}")

# Summary statistics
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)

print(f"\nMean IoU: {mean_iou:.4f}")
print(f"Mean Dice Coefficient: {mean_dice:.4f}")
