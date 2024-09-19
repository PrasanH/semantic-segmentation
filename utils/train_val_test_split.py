import os
import shutil
import random
from typing import Tuple

def create_output_folders(output_dir: str) -> None:
    """Create train, test, and validation folders with images and labels subfolders."""
    for split in ['train', 'test', 'valid']:
        split_path = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_path, 'labels'), exist_ok=True)

def pair_images_and_labels(images_dir: str, labels_dir: str) -> list:
    """Pair image and label files based on their common timestamp identifier."""
    images = sorted([img for img in os.listdir(images_dir) if img.endswith('.png')])
    labels = sorted([lbl for lbl in os.listdir(labels_dir) if lbl.endswith('.png')])

    # Assuming the identifier (timestamp) is the first part of the filename
    paired_files = []
    for image in images:
        identifier = image.split('_')[0]  # Example: '1721396432'
        label = next((lbl for lbl in labels if lbl.startswith(identifier)), None)
        if label:
            paired_files.append((image, label))
    
    return paired_files

def split_dataset(pairs: list, train_split: float, test_split: float) -> Tuple[list, list, list]:
    """Split the dataset into train, test, and validation sets."""
    random.shuffle(pairs)
    
    train_size = int(train_split * len(pairs))
    test_size = int(test_split * len(pairs))
    
    train_pairs = pairs[:train_size]
    test_pairs = pairs[train_size:train_size + test_size]
    valid_pairs = pairs[train_size + test_size:]

    return train_pairs, test_pairs, valid_pairs

def copy_files(pairs: list, source_images_dir: str, source_labels_dir: str, dest_dir: str, split: str) -> None:
    """Copy paired images and labels to the appropriate split folder (train, test, valid)."""
    for image, label in pairs:
        shutil.copy(os.path.join(source_images_dir, image), os.path.join(dest_dir, split, 'images', image))
        shutil.copy(os.path.join(source_labels_dir, label), os.path.join(dest_dir, split, 'labels', label))

def organize_dataset(images_dir: str, labels_dir: str, output_dir: str, train_split: int, test_split: int, valid_split: int) -> None:
    """Organize dataset by creating splits and copying files."""
    #if train_split + test_split + valid_split != 1.0:
        #raise ValueError("The splits must sum to 1")
    
    create_output_folders(output_dir)

    pairs = pair_images_and_labels(images_dir, labels_dir)
    
    train_pairs, test_pairs, valid_pairs = split_dataset(pairs, train_split, test_split)
    
    # Copy the files into respective folders
    copy_files(train_pairs, images_dir, labels_dir, output_dir, 'train')
    copy_files(test_pairs, images_dir, labels_dir, output_dir, 'test')
    copy_files(valid_pairs, images_dir, labels_dir, output_dir, 'valid')

    print(f"Dataset organized successfully with {len(train_pairs)} train pairs, {len(test_pairs)} test pairs, and {len(valid_pairs)} validation pairs.")

# Example usage:
# organize_dataset('/path/to/images', '/path/to/labels', '/path/to/output', 0.7, 0.2, 0.1)
organize_dataset('./data/unreal_images/train/images/', './data/unreal_images/train/labels/', './data/unreal_images/split', 0.70, 0.10, 0.20)