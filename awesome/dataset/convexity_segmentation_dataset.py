

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from awesome.dataset.prior_dataset import PriorDataset, prior
from awesome.dataset.subdivisible_dataset import NOSUBSET, SubdivisibleDataset
from awesome.dataset.torch_datasource import TorchDataSource
import re 
from torch import Tensor
import pandas as pd
import os
import logging
from PIL import Image
import numpy as np
from enum import Enum

from awesome.util.temporary_property import TemporaryProperty

class OutputMode(Enum):
    PIXEL = 1
    IMAGE = 2

class ConvexitySegmentationDataset(SubdivisibleDataset, PriorDataset):

    IMAGE_PATTERN = r"img_(?P<index>\d+).png"

    GROUND_TRUTH_PATTERN = r"GT_(?P<index>\d+).png"

    SCRIBBLES_PATTERN = r"scribbleMask_(?P<index>\d+).png"

    __index__: pd.DataFrame
    """The index of images to load."""

    def __init__(self, 
                 dataset_path: str, 
                 scribble_percentage: float = 1.,
                 dtype: torch.dtype = torch.float32, 
                 subset: Optional[Union[int, List[int], slice]] = None,
                 **kwargs) -> None:
        super().__init__(returns_index=False, **kwargs)
        self.dataset_path = dataset_path
        self.__index__ = self.index()
        self.dtype = dtype
        self.output_mode: OutputMode = OutputMode.PIXEL
        self.scribble_percentage = scribble_percentage

    def create_subset_mapping(self) -> Dict[int, int]:
        index = self.__index__.reset_index()
        subset = index.iloc[self.__subset_specifier__]
        if isinstance(subset, pd.Series):
            return {0: subset.name}
        return {i: v for i, v in enumerate(subset.index)}

    def index(self) -> pd.DataFrame:
        # Getting images
        image_dir = os.path.normpath(os.path.join(self.dataset_path, "img"))
        res = dict()

        for file in os.listdir(image_dir):
            m = re.fullmatch(ConvexitySegmentationDataset.IMAGE_PATTERN, file)
            if m is not None:
                index = int(m.group("index"))
                image_path = os.path.join(image_dir, file)
                if index not in res:
                    res[index] = dict()
                    res[index]["image_path"] = image_path
        
        # Getting ground truth
        gt_dir = os.path.normpath(os.path.join(self.dataset_path, "ground_truth"))
        for file in os.listdir(gt_dir):
            m = re.fullmatch(ConvexitySegmentationDataset.GROUND_TRUTH_PATTERN, file)
            if m is not None:
                index = int(m.group("index"))
                gt_path = os.path.join(gt_dir, file)
                if index not in res:
                    logging.warning(f"Ground truth image {gt_path} has no corresponding image!")
                    res[index] = dict()
                res[index]["gt_path"] = gt_path
        
        # Getting scribbles
        scribble_dir = os.path.normpath(os.path.join(self.dataset_path, "user_scribbles"))
        for file in os.listdir(scribble_dir):
            m = re.fullmatch(ConvexitySegmentationDataset.SCRIBBLES_PATTERN, file)
            if m is not None:
                index = int(m.group("index"))
                scribble_path = os.path.join(scribble_dir, file)
                if index not in res:
                    logging.warning(f"Scribble image {scribble_path} has no corresponding image!")
                    res[index] = dict()
                res[index]["scribble_path"] = scribble_path
    
        df = pd.DataFrame.from_dict(res, orient="index")
        df = df.sort_index().reset_index(drop=False)
        df = df.rename(columns={"index": "image_index"})
        return df
            
    @prior()
    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]]:
        """Returns the item at the given index.

        Parameters
        ----------
        index : int
            Index of the image within the dataloader. 

        Returns
        -------
        Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]
            Returning a tuple containing
            1. Tuple of (Input-Tensor (RGB Image), Foreground-Mask-Tensor (Binary mask), Background-Mask-Tensor (Binary mask))
            2. Ground truth tensor (Binary mask)

        OR 

        Tuple[Tensor, Tensor]
            Returning a tuple containing
            1. Test pixels (Tensor of shape (N, 5) where N is the number of important pixels)
            2. Ground truth pixels (Tensor of shape (N, 5) where N is the number of important pixels)
            Importance is wether the pixel is defined withhin a mask.
            The last dimension is the YX Coordinate + RGB value of the pixel.

        """        
        row = self.__index__.iloc[self.get_data_index(index)]
        image_path = row["image_path"]
        gt_path = row["gt_path"]
        scribble_path = row["scribble_path"]
        
        img = self.load_image(image_path)
        gt = self.load_ground_truth(gt_path)

        background_mask, foreground_mask = self.get_masks(scribble_path)
        
        if self.output_mode == OutputMode.PIXEL:
            test_px = ConvexitySegmentationDataset.get_important_pixels(img, foreground_mask, background_mask)
            gt_px = ConvexitySegmentationDataset.get_gt_important_pixels(gt, foreground_mask, background_mask)
            if self.scribble_percentage < 1.:
                _random_px = self.get_random_pix(img, test_px.shape[-2])
                test_px = torch.cat([test_px, _random_px], dim=-2)
            return test_px, gt_px
        elif self.output_mode == OutputMode.IMAGE:
            return (img, foreground_mask, background_mask), gt
 
    def get_image(self, index: int) -> torch.Tensor:
        with TemporaryProperty(self, output_mode=OutputMode.IMAGE, return_prior=False):
            return self[index][0][0]

    def get_random_pix(self, image, interesting_pixels: int) -> torch.Tensor:
        torch.manual_seed(self.split_seed)
        random_pixels = (interesting_pixels * (1 / self.scribble_percentage)) - interesting_pixels
        random_pixels = int(math.ceil(random_pixels))

        needed_pixels = random_pixels
        mask = torch.zeros_like(image[0, ...], dtype=torch.bool)
        indices = torch.argwhere(mask)

        max_iter = 1000
        i = 0
        # Looping to be sure to draw enough
        while needed_pixels > 0 and (i < max_iter):
            random_indices_y = np.random.randint(0, image.shape[-2], size=needed_pixels)
            random_indices_x = np.random.randint(0, image.shape[-1], size=needed_pixels)
            random_indices = torch.tensor(np.stack([random_indices_y, random_indices_x]))
            mask[random_indices[0], random_indices[1]] = True
            indices = torch.argwhere(mask)
            needed_pixels = random_pixels - indices.shape[0]
            i+=1

        if needed_pixels > 0:
            raise ValueError(f"Could not draw enough random pixels after {max_iter} iterations.")
        
        radom_rgb = image[..., mask]
        indices = torch.argwhere(mask)
        norm_indices = indices / torch.tensor(image.shape[-2:]) - 0.5
        return torch.cat([norm_indices, radom_rgb.T], dim=-1)

    @staticmethod
    def get_important_pixels(image: torch.Tensor, foreground_mask: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        if foreground_mask is not None and background_mask is not None:
            interesting_pixels = torch.logical_or(foreground_mask, background_mask)
        elif foreground_mask is not None:
            interesting_pixels = foreground_mask
        elif background_mask is not None:
            interesting_pixels = background_mask
        else:
            interesting_pixels = torch.ones(image.shape[1:], dtype=torch.bool)
        rgb_values = image[..., interesting_pixels]
        indices = torch.argwhere(interesting_pixels)
        norm_indices = indices / torch.tensor(image.shape[-2:]) - 0.5
        return torch.cat([norm_indices, rgb_values.T], dim=-1)
    
    @staticmethod
    def get_gt_important_pixels(gt_mask: torch.Tensor, foreground_mask: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        if foreground_mask is not None and background_mask is not None:
            interesting_pixels = torch.logical_or(foreground_mask, background_mask)
        elif foreground_mask is not None:
            interesting_pixels = foreground_mask
        elif background_mask is not None:
            interesting_pixels = background_mask
        else:
            interesting_pixels = torch.ones_like(gt_mask, dtype=torch.bool)

        _class = gt_mask[..., interesting_pixels][..., None]
        
        return _class

    def load_ground_truth(self, gt_path: str) -> Tensor:
        img_pil = Image.open(gt_path)
        img = np.array(img_pil, dtype='float')/255.0
        img = np.where(img == 1, 0., 1.)
        return torch.from_numpy(img).to(dtype=torch.float32)

    def load_image(self, image_path: str) -> Tensor:
        img_pil = Image.open(image_path)
        img = np.array(img_pil, dtype='float')/255.0
        img = img[:,:,0:3]
        return torch.from_numpy(img.transpose(2, 0, 1)).to(dtype=self.dtype)

    def get_masks(self, image_path: str):
        img_pil=Image.open(image_path)
        img= np.array(img_pil, dtype='float')/255.0
        no_mask, bg, fg = np.unique(img) # Assuming order of: No segmentation = 0, Background = 1, Foreground = 2
        bg = torch.from_numpy(np.where(img == bg, 1., 0.)).to(dtype=torch.bool)
        fg = torch.from_numpy(np.where(img == fg, 1., 0.)).to(dtype=torch.bool)
        return bg, fg

    def __len__(self) -> int:
        sub = self.subset_len()
        if sub == NOSUBSET:
            return len(self.__index__)
        return sub

    def decode_encoding(self, output: torch.Tensor) -> torch.Tensor:
        """Decodes the output of the model to the original class labels.

        Parameters
        ----------
        output : torch.Tensor
            Output of the model.

        Returns
        -------
        torch.Tensor
            Decoded output.
        """
        return output >= 0.5 # 0 is foreground; 1 is background
    