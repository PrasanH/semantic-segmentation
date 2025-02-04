import os
from PIL import Image
import torch
import numpy as np
from collections import Counter

"""
Uses CPU
Calculates the count of pixel values belonging to each class.
Useful for training with weighted class weights

"""


# Define RGB values for each class
sky = [0, 149, 200]
obstacle = [[120, 187, 255], [136, 97, 0], [158, 158, 158], [165, 63, 0], [136, 97, 0], 
            [31, 31, 31], [32, 32, 32], [131, 131, 131], [132, 132, 132], [169, 0, 45], 
            [176, 176, 176]]
vegetation = [120, 113, 0]
landscape_terrain = [228, 196, 80]

# Define a function to map RGB to class labels
def rgb_to_class(pixel):
    if pixel.tolist() == sky:
        return 0  # sky
    elif pixel.tolist() in obstacle:
        return 1  # obstacle
    elif pixel.tolist() == vegetation:
        return 2  # vegetation
    elif pixel.tolist() == landscape_terrain:
        return 3  # landscape_terrain
    else:
        return None  # Ignore or handle unknown values as needed

# Path to directory containing mask images
image_dir = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/train/labels/"  # Replace with your actual path

# Initialize counts for each class
class_counts = Counter()

# Loop through each image file in the directory
for filename in os.listdir(image_dir):
    print('Please wait.WIP.......')
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust for your image format
        image_path = os.path.join(image_dir, filename)
        
        # Load image
        mask_image = Image.open(image_path).convert("RGB")
        mask = np.array(mask_image)  # Convert to NumPy array for easier pixel access

        # Flatten mask and iterate over pixels
        for pixel in mask.reshape(-1, 3):
            class_label = rgb_to_class(pixel)
            if class_label is not None:
                class_counts[class_label] += 1

# Calculate class weights for CrossEntropyLoss
total_pixels = sum(class_counts.values())
class_weights = torch.tensor([total_pixels / class_counts[i] for i in range(4)], dtype=torch.float32)

print("Class Counts:", class_counts)
print("Class Weights:", class_weights)
