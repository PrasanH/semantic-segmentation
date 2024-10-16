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

    Attributes:
        root: (str)
            the root directory (e.g., 'train/')
        transforms: (Optional[Callable])
            torch transforms to use
        image_names: List of image paths
        mask_names: List of corresponding mask paths
    """

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

        image_dir = os.path.join(root, "images")
        label_dir = os.path.join(root, "labels")

        # Collect all the image paths
        self.image_names = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith("_visible.png")
            ]
        )

        # Collect corresponding mask paths
        self.mask_names = [
            os.path.join(label_dir, f.replace("_visible.png", "_class.png"))
            for f in os.listdir(image_dir)
            if f.endswith("_visible.png")
        ]

        if resize_shape:
            self.image_height, self.image_width = resize_shape
            self.resize = True
        else:
            self.image_height, self.image_width = (544, 1024)
            self.resize = False

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

        image = Image.open(image_path)
        image = image.convert("RGB")

        # Load the mask
        mask = Image.open(mask_path)
        mask = mask.convert("RGB")  # Ensure the mask is in RGB format
        mask = np.array(mask)

        # Resize the mask if needed
        if self.resize:
            mask = cv2.resize(
                mask,
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        
        sky = [0, 149,200]
        obstacle = [[120,187,255], [136,97,0],[158,158,158] ,[165,63,0],[136, 97, 0] , [31,31,31],[32,32,32],[131,131,131],[132,132,132],[169,0,45],[176,176,176]]
        vegetation = [120,113,0]
        landscape_terrain = [228,196,80]
        num_classes = 4

        one_hot_mask = np.zeros((num_classes, self.image_height, self.image_width), dtype=np.float32)


        # Helper function to match RGB values
        def match_category(rgb_values, class_id):
            for rgb in rgb_values:
                mask_match = np.all(mask == rgb, axis=-1)
                one_hot_mask[class_id][mask_match] = 1

        # Map RGB values to classes
        match_category([sky], 0)
        match_category(obstacle, 1)
        match_category([vegetation], 2)
        match_category([landscape_terrain], 3)
    
        # Any remaining pixels (not categorized) are considered obstacles
        uncategorized_pixels = np.all(one_hot_mask == 0, axis=0)
        one_hot_mask[1][uncategorized_pixels] = 1
        
        sample = {"image": image, "mask": one_hot_mask}

        # Apply any transformations if provided
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample["mask"] = torch.as_tensor(sample["mask"], dtype=torch.uint8)


        #print('sample shape', sample['mask'].shape)
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
    print("image_datasets", image_datasets)
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, drop_last=False
        )
        for x in ["train", "valid"]
    }
    return dataloaders
