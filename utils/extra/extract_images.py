import os
import shutil

def extract_and_rename_images(source_directory, images_directory, labels_directory):
    # Create the destination directories if they don't exist
    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(labels_directory, exist_ok=True)

    # Get a list of all folder names in the source directory
    folder_list = sorted([d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))])
    
    for idx, folder in enumerate(folder_list):
        folder_path = os.path.join(source_directory, folder)
        
        # Define source paths for rgb.png and label.png
        rgb_source = os.path.join(folder_path, 'rgb.jpg')
        label_source = os.path.join(folder_path, 'labels.png')

        # Create target filenames
        rgb_target = os.path.join(images_directory, f"{str(idx).zfill(4)}.png")
        label_target = os.path.join(labels_directory, f"{str(idx).zfill(4)}_label.png")

        # Move the files to the corresponding folders with new names
        if os.path.exists(rgb_source):
            shutil.copy(rgb_source, rgb_target)
        if os.path.exists(label_source):
            shutil.copy(label_source, label_target)

        print(f"Processed {folder}: Saved rgb as {rgb_target}, label as {label_target}")

# Example usage:
source_dir = r"H:\semantic_segmentation\semantic-segmentation\data\yamaha_v0\train"  # Replace with the path to your source folder containing iid0000, iid0001, etc.
images_dir = r"H:\semantic_segmentation\semantic-segmentation\data\yamaha_sorted\images"          # Replace with the path to your destination 'images' folder
labels_dir = r"H:\semantic_segmentation\semantic-segmentation\data\yamaha_sorted\labels"          # Replace with the path to your destination 'labels' folder

extract_and_rename_images(source_dir, images_dir, labels_dir)
