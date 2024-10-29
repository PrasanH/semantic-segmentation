import os
from PIL import Image
import torch
import torchvision.transforms as T
from collections import Counter

# Define RGB values for each class
sky = torch.tensor([0, 149, 200], device='cuda')
'''
obstacle = torch.tensor([[120, 187, 255], [136, 97, 0], [158, 158, 158], [165, 63, 0], 
                         [136, 97, 0], [31, 31, 31], [32, 32, 32], [131, 131, 131], 
                         [132, 132, 132], [169, 0, 45], [176, 176, 176]], device='cuda')
'''
vegetation = torch.tensor([120, 113, 0], device='cuda')
landscape_terrain = torch.tensor([228, 196, 80], device='cuda')

# Mapping function using CUDA tensors
def rgb_to_class_tensor(mask_tensor):
    class_mask = torch.full(mask_tensor.shape[:2], -1, dtype=torch.int, device='cuda')  # Start with -1 for uncategorized pixels

    # Assign each class based on pixel values
    class_mask[(mask_tensor == sky).all(dim=-1)] = 0  # Sky
    class_mask[(mask_tensor == vegetation).all(dim=-1)] = 2  # Vegetation
    class_mask[(mask_tensor == landscape_terrain).all(dim=-1)] = 3  # Landscape

    # Any remaining pixels (still -1) are considered obstacles
    class_mask[class_mask == -1] = 1  # Obstacle class

    return class_mask

# Path to directory containing mask images
image_dir = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/train/labels/"   

# Initialize counts for each class
class_counts = torch.zeros(4, dtype=torch.int64, device='cuda')

# Image transformation
transform = T.Compose([
    T.ToTensor(),  # Convert to tensor
    T.Lambda(lambda x: x.permute(1, 2, 0) * 255)  # Change to HxWxC and scale to 0-255
])

# Process images on the GPU
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust for your image format
        image_path = os.path.join(image_dir, filename)
        
        # Load image, transform, and move to CUDA
        mask_image = Image.open(image_path).convert("RGB")
        mask_tensor = transform(mask_image).to(torch.uint8).to('cuda')

        # Map to class labels and count each class
        class_mask = rgb_to_class_tensor(mask_tensor)
        for i in range(4):  # Assuming 4 classes
            class_counts[i] += (class_mask == i).sum()

# Calculate class weights for CrossEntropyLoss
total_pixels = class_counts.sum().float()
class_weights = total_pixels / class_counts.float()
normalized_class_weights = class_weights / class_weights.sum()

# Print class counts and normalized weights
print("Class Counts:", class_counts.cpu().numpy())
print("Normalized Class Weights:", normalized_class_weights.cpu().numpy())
print("Class Counts:", class_counts.cpu().numpy())
print("Class Weights:", class_weights.cpu().numpy())


'''
Class Counts: [266803673   4339813 697704760 344978774]
Class Weights: [  4.924321  302.7382      1.8830702   3.8084285]

Class Weights normalized : [  25  50      15   10]
'''