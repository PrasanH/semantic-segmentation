""" YamahaCMU Dataloaders"""

import glob
from typing import Any, Callable, Optional

import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import numpy as np
import cv2
from PIL import Image

import os
import logging
from datetime import datetime

'''
class YamahaCMUDataset(VisionDataset):
    """ A class that represents the Yamaha-CMU Off-Road dataset

    Attributes:
        root: (str)
            the root directory
        transforms: (Optional[Callable])
            torch transforms to use

    Methods:
        __len__():
            returns the length of the dataset
        __getitem__(index):
            returns the item at the given index of this dataset
    """

    def __init__(self, root: str, resize_shape: tuple,
                 transforms: Optional[Callable] = None) -> None:
        """ Initializes a YamahaCMUDataset object

        Args:
            root: (str)
                the root directory
            transforms: (Optional[Callable])
                torch transforms to use
        """
        super().__init__(root, transforms)
        image_paths = []
        mask_paths = []
        image_mask_pairs = glob.glob(root + '/*/')
        for image_mask in image_mask_pairs:
            image_paths.append(glob.glob(image_mask + '*.jpg')[0])
            mask_paths.append(glob.glob(image_mask + '*.png')[0])
        self.image_names = image_paths
        self.mask_names = mask_paths

        if resize_shape:
            self.image_height, self.image_width = resize_shape
            self.resize = True
        else:
            self.image_height, self.image_width = (544, 1024)
            self.resize = False

    def __len__(self) -> int:
        """ Returns the length of the dataset """
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        """ Returns the item at the given index of this dataset

        Args:
            index: (int)
                the index of the item to get

        Returns:
            the sample at the given index
        """
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        image = Image.open(image_path)
        image = image.convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        class_colors = np.unique(mask)
        if self.resize:
            mask = cv2.resize(mask,
                              dsize=(self.image_width, self.image_height),
                              interpolation=cv2.INTER_CUBIC)
        # remove void class (atv)
        if 0 in class_colors:
            class_colors = class_colors[1:]
        label_masks = mask == class_colors[:, None, None]
        masks = np.zeros((8, self.image_height, self.image_width))
        for index, class_color in enumerate(class_colors):
            masks[class_color - 1] = label_masks[index, :, :] * 255
        sample = {"image": image, "mask": masks}
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample['mask'] = torch.as_tensor(sample['mask'], dtype=torch.uint8)
        return sample


'''


class UnrealDataset(VisionDataset):
    """A class that represents the Unreal engine offroad dataset
    Classes:
        0: Sky [0, 149, 200]
        1: Obstacles (anything not explicitly classified as sky, vegetation, or landscape)
        2: Vegetation [120, 113, 0]
        3: Landscape Terrain [228, 196, 80]

    Attributes:
        root: (str)
            the root directory (e.g., 'train/')
        transforms: (Optional[Callable])
            torch transforms to use
        image_names: List of image paths
        mask_names: List of corresponding mask paths
        logger: logging.Logger
            Logger instance for tracking dataset operations
    """
    class_colors = {
        'sky': [0, 149, 200],
        'vegetation': [120, 113, 0],
        'landscape': [228, 196, 80]
    }

    def __init__(
        self,
        root: str,
        resize_shape: tuple = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        """Initializes a Unreal engine dataset object

        Args:
            root: (str)
                the root directory containing 'images' and 'labels' folders
            resize_shape: (tuple, optional)
                the target size to resize the images and masks
            transforms: (Optional[Callable])
                torch transforms to apply
        """
        super().__init__(root, transforms)

        # Setup logging
        #log_dir = os.path.join(os.path.dirname(root), 'logs')
        #os.makedirs(log_dir, exist_ok=True)
        
        log_dir = 'logs'   # directory for logging loss df
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'dataset_loading_{timestamp}.log')

        self.logger = logging.getLogger(f'UnrealDataset_{timestamp}')
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        #console_handler = logging.StreamHandler()
        #console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        #console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        #self.logger.addHandler(console_handler)
        
        self.logger.info(f"Initializing dataset from root: {root}")

        image_dir = os.path.join(root, "images")
        label_dir = os.path.join(root, "labels")

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith("_visible.png")
        ])

        self.image_names = []
        self.mask_names = []
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            mask_file = img_file.replace("_visible.png", "_class.png")
            mask_path = os.path.join(label_dir, mask_file)
            
            # Verify mask exists
            if not os.path.exists(mask_path):
                self.logger.warning(f"Missing mask file for {img_file}: {mask_path}")
                continue
                
            self.image_names.append(img_path)
            self.mask_names.append(mask_path)

        self.logger.info(f"Found {len(self.image_names)} valid image-mask pairs")

        if resize_shape:
            self.image_height, self.image_width = resize_shape
            self.resize = True
            self.logger.info(f"Images will be resized to {resize_shape}")
        else:
            self.image_height, self.image_width = (544, 1024)
            self.resize = False
            self.logger.info("Using default image size: (544, 1024)")


    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.image_names)


    def __getitem__(self, index: int) -> dict:
        """Returns the item at the given index of this dataset

        Args:
            index: (int)
                the index of the item to get

        Returns:
            dict: A dictionary with 'image' and 'mask' keys
        """
        # Load the image
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        
        self.logger.info(f"Loading image-mask pair {index}:")
        self.logger.info(f"  Image: {os.path.basename(image_path)}")
        self.logger.info(f"  Mask:  {os.path.basename(mask_path)}")

        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

        try:
            # Load the mask
            mask = Image.open(mask_path)
            mask = mask.convert("RGB")  # Ensure the mask is in RGB format
            mask = np.array(mask)
        except Exception as e:
            self.logger.error(f"Error loading mask {mask_path}: {str(e)}")
            raise

        # Resize the mask if needed
        if self.resize:
            mask = cv2.resize(
                mask,
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_NEAREST,
            )

        num_classes = 4
        one_hot_mask = np.zeros((num_classes, self.image_height, self.image_width), dtype=np.float32)

        # Initialize pixel counts for logging
        pixel_counts = {
            'sky': 0,
            'vegetation': 0,
            'landscape': 0,
            'obstacle': 0
        }

        # First classify the explicitly defined classes
        for class_idx, (class_name, rgb) in enumerate([
            ('sky', self.class_colors['sky']),
            (None, None),  # Skip index 1 (obstacles)
            ('vegetation', self.class_colors['vegetation']),
            ('landscape', self.class_colors['landscape'])
        ]):
            if class_name is not None:  # Skip the obstacle class (index 1)
                mask_match = np.all(mask == rgb, axis=-1)
                one_hot_mask[class_idx][mask_match] = 1
                pixel_counts[class_name] = np.sum(mask_match)

        # Now classify all remaining pixels as obstacles (class index 1)
        is_obstacle = np.all(one_hot_mask[[0, 2, 3]] == 0, axis=0)
        one_hot_mask[1][is_obstacle] = 1
        pixel_counts['obstacle'] = np.sum(is_obstacle)

        # Log pixel distribution
        total_pixels = self.image_height * self.image_width
        self.logger.info(f"Pixel distribution for {os.path.basename(mask_path)}:")
        for class_name, count in pixel_counts.items():
            percentage = (count / total_pixels) * 100
            self.logger.info(f"  {class_name}: {count} pixels ({percentage:.2f}%)")
        
        sample = {"image": image, "mask": one_hot_mask}

        # Apply any transformations if provided
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample["mask"] = torch.as_tensor(sample["mask"], dtype=torch.float32)

        return sample


def get_dataloader(
    data_dir: str, batch_size: int = 2, resize_shape: tuple = None
) -> torch.utils.data.DataLoader:
    """Creates a dataloader for the given dataset

    Args:
        data_dir: (str)
            the directory of the dataset
        batch_size: (int=2)
            the batch size to use

    Returns:
        torch.utils.data.DataLoader
    """

    if resize_shape:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(resize_shape),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # image_datasets = {x: YamahaCMUDataset(data_dir + x, resize_shape, transforms=preprocess) for x in ['train', 'valid']}

    image_datasets = {
        x: UnrealDataset(data_dir + x, resize_shape, transforms=preprocess)
        for x in ["train", "valid"]
    }
    #print("image_datasets", image_datasets)
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, drop_last=False
        )
        for x in ["train", "valid"]
    }
    return dataloaders
