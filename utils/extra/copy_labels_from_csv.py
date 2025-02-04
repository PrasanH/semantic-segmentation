import os
import shutil
import csv


"""
scans a folder of images. and then compares with a .csv file (contains test image names). from this csv, it extarcts basename and 
if it is found in  source folder, it will move them to a different folder. 

"""
# Paths
csv_file = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/validation_images.csv"
source_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/ienet2_380k/"
destination_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/Nov_24/ienet2_380k/valid"  # Folder to move the matching images

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read the base names (before "_Camera0_visible.png") from the CSV file
with open(csv_file, mode="r") as csvfile:
    csv_reader = csv.reader(csvfile)
    # Skip the header
    next(csv_reader)
    # Extract the base names
    base_names = {row[0].split("_")[0] for row in csv_reader}

# Loop through the files in the source folder
for image_file in os.listdir(source_folder):
    # Extract the base name from the current file
    base_name = os.path.basename(image_file).split("_")[0]
    # Check if the base name matches any entry in the CSV
    if base_name in base_names:
        # Move the file to the destination folder
        src_path = os.path.join(source_folder, image_file)
        dst_path = os.path.join(destination_folder, image_file)
        shutil.move(src_path, dst_path)

print(f"Moved matching images to {destination_folder}")
