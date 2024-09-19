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
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        # Resize the mask if needed
        if self.resize:
            mask = cv2.resize(
                mask,
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        
        sky = [142]
        rock = [147, 148]
        vegetation = [174]
        landscape_terrain = list(range(94, 103))
        wall = [191]
        vehicle = list(range(175, 186))
        tree_trunk = [158]
        furniture = list(range(57, 94))
        barn = [43, 144, 145, 146]
        building = list(range(31, 37))
        roadside_object = list(range(130, 142))

        defined_categories = set(sky + rock + vegetation + landscape_terrain + wall + vehicle + tree_trunk + furniture + barn + building + roadside_object)
        all_labels = set(range(195))
        unlabeled = list(all_labels - defined_categories)

        # Initialize a shader_map-like array for masks (12 classes)
        masks = np.zeros((12, self.image_height, self.image_width), dtype=np.float32)

        # Populate the masks array
        masks[0] = np.isin(mask, sky).astype(np.float32)
        masks[1] = np.isin(mask, rock).astype(np.float32)
        masks[2] = np.isin(mask, vegetation).astype(np.float32)
        masks[3] = np.isin(mask, landscape_terrain).astype(np.float32)
        masks[4] = np.isin(mask, wall).astype(np.float32)
        masks[5] = np.isin(mask, vehicle).astype(np.float32)
        masks[6] = np.isin(mask, tree_trunk).astype(np.float32)
        masks[7] = np.isin(mask, furniture).astype(np.float32)
        masks[8] = np.isin(mask, barn).astype(np.float32)
        masks[9] = np.isin(mask, building).astype(np.float32)
        masks[10] = np.isin(mask, roadside_object).astype(np.float32)
        masks[11] = np.isin(mask, unlabeled).astype(np.float32)
        # Prepare the sample
        sample = {"image": image, "mask": masks}

        # Apply any transformations if provided
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample["mask"] = torch.as_tensor(sample["mask"], dtype=torch.uint8)

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
