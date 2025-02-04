import os
import shutil


"""
Copies the labels of test images from a .txt/csv file file from epe . 
this file contains 4 paths per line where the last line is the path for image label (see example below)
Please check the num_channels after copying. If its 4, remove the alpha channel. (code is in epe utils folder)

example : fake_unreal/images/1721395773_Camera0_visible.png,robust_labels/,.gbuffers/1721395773.npz,/gt_labels/1721395773_Camera0_class.png

These paths are relative paths. So, we will also define a root path below, which helps in copying the files. 

"""

# Define the path to your .txt file and the destination folder
txt_file_path = "/home/pdhegde/EPE_thesis/epe_thesis/code/data_forest_3/fake_unreal/test.txt"
destination_folder = "/home/pdhegde/semseg_git_fork/semantic-segmentation/data/test_labels_simulated/images/"


os.makedirs(destination_folder, exist_ok=True)

# Open the .txt file and process each line
with open(txt_file_path, 'r') as file:
    for line in file:
        # Split the line by commas and get the second path
        paths = line.strip().split(',')
        second_path = paths[3]

        root_path = "/home/pdhegde/EPE_thesis/epe_thesis/code/"

        second_path = os.path.join(root_path,second_path)

        second_path = second_path.replace('gt_labels', 'gt_labels_colour')

        
        
        # Copy the file to the destination folder
        if os.path.exists(second_path):
            shutil.copy(second_path, destination_folder)
        else:
            print(f"File not found: {second_path}")


