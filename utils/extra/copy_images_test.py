import os
import random
import csv

# Folder containing the images
images_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/labels_for_the_fakes_1500/"  # Replace with the path to your images folder

# Output CSV file path
output_csv = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/test_images.csv"

# Number of images to select
num_images_to_select = 150

# Fetch all image names ending with "_Camera0_class.png"
image_names = [
    f for f in os.listdir(images_folder)
    if f.endswith("_Camera0_class.png") and os.path.isfile(os.path.join(images_folder, f))
]

# Replace "_Camera0_class.png" with "_Camera0_visible.png"
modified_image_names = [f.replace("_Camera0_class.png", "_Camera0_visible.png") for f in image_names]

# Randomly select 150 images
selected_images = random.sample(modified_image_names, min(num_images_to_select, len(modified_image_names)))

# Write the selected image names to a CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(["Image Name"])
    # Write selected image names
    for image_name in selected_images:
        csv_writer.writerow([image_name])

print(f"Randomly selected {len(selected_images)} images saved to {output_csv}")

