'''
Copy image labels for the corresponding images which we already have to a given folder

'''


import os
import shutil

# Define your folder paths
all_images_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images_all/train/images/'  # Folder where the images are stored (original images)
all_labels_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images_all/train/labels/'  # Folder where the labels are stored (original labels)


images_subset_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/simulated/images/'  # Folder where the selected images are copied (new images)
labels_subset_folder = '/home/pdhegde/semseg_git_fork/semantic-segmentation/data/unreal_images/simulated/labels/'  # Folder where the corresponding labels should be copied (new labels)

# Get the list of images in images_subset_folder
image_files = os.listdir(images_subset_folder)



# Iterate over the files in images_subset_folder and copy the corresponding labels from all_labels_folder to labels_subset_folder
for image_file in image_files:
    if image_file.endswith('_visible.png'):
        # Extract the base filename (remove the "_visible.png" part)
        base_name = image_file.replace('_visible.png', '')
        
        # Construct the corresponding label filename
        label_file = f"{base_name}_class.png"
        
        # Check if the label exists in folder 2
        label_path = os.path.join(all_labels_folder, label_file)
        if os.path.exists(label_path):
            # Copy the label to folder 4
            shutil.copy(label_path, labels_subset_folder)
            print(f"Copied {label_file} to {labels_subset_folder}")
        else:
            print(f"Label {label_file} not found in {all_labels_folder}")

print("Label copying complete.")


