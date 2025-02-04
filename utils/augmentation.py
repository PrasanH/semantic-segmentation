import os
from PIL import Image
import numpy as np
import albumentations as A
import random

"""
Carries out augmentation to the images and the corresponding labels in the image and label folder defined 

Check what all augmnetations you need for your use in the function get_augmentation_pipeline()

the code adds a randomly generated number as a subscript to the augmented image and labels (same for the corresponding pairs)

also, adds _Camera0_visible for images and _Camera0_class for the label images 

Define your input and output folders at the end of this file

"""


# Define the augmentation pipeline using Albumentations
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.Rotate(limit=(180, 180), p=0.5),  # Rotate by 180 degrees only
        #A.Transpose(p=0.5),       # Random transpose of the image
        #A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast adjustment
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=(180,180), p=0.5)  # Random shift, scale, and rotation
    ], additional_targets={'label': 'image'})

# Function to load images and labels, perform augmentation, and save results
def augment_images_with_labels(image_folder, label_folder, output_image_folder, output_label_folder):
    # Ensure output folders exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    # Get augmentation pipeline
    augmentation_pipeline = get_augmentation_pipeline()

    # Loop over all image files in the folder
    for filename in os.listdir(image_folder):
        if "_Camera0_visible.png" in filename:
            # Derive the corresponding label filename
            base_name = filename.split('_Camera0_visible.png')[0]
            label_filename = f"{base_name}_Camera0_class.png"

            # Paths to image and label
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, label_filename)
            
            # Check if the corresponding label exists
            if not os.path.exists(label_path):
                print(f"Label for {filename} not found, skipping...")
                continue

            # Load image and label
            image = np.array(Image.open(image_path))
            label = np.array(Image.open(label_path))

            # Apply the same augmentation to both image and label
            augmented = augmentation_pipeline(image=image, label=label)

            # Extract augmented image and label
            augmented_image = Image.fromarray(augmented['image'])
            augmented_label = Image.fromarray(augmented['label'])

            # Generate a random number to be used for both the image and label filenames
            random_number = random.randint(1000, 9999)
            new_filename = f"{base_name}_aug_{random_number}_Camera0_visible.png"
            new_label_filename = f"{base_name}_aug_{random_number}_Camera0_class.png"

            # Save the augmented image and label
            augmented_image.save(os.path.join(output_image_folder, new_filename))
            augmented_label.save(os.path.join(output_label_folder, new_label_filename))
            print(f"Augmented {filename} and saved as {new_filename}")

# Example usage
image_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/ienet2_380k/train/'
label_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/labels_for_the_fakes_1500/train/'

#image_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/nov_24_img_labels/train/images/'
#label_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/nov_24_img_labels/train/labels'


output_image_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/nov_24_training/ienet2_380k/train/images/'
output_label_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/nov_24_training/ienet2_380k/train/labels'

augment_images_with_labels(image_folder, label_folder, output_image_folder, output_label_folder)
