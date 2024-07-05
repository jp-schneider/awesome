
import logging
import math
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

import torch
from awesome.dataset.image_sample import ImageSample
from awesome.dataset.prior_dataset import PriorDataset
from awesome.dataset.subdivisible_dataset import NOSUBSET, SubdivisibleDataset
from awesome.dataset.torch_datasource import TorchDataSource

from awesome.dataset.prior_dataset import PriorDataset, prior
from tqdm.autonotebook import tqdm

from awesome.run.awesome_config import AwesomeConfig
from awesome.util.matplotlib import saveable
from awesome.util.temporary_property import TemporaryProperty
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes


class AwesomeDataset(SubdivisibleDataset, PriorDataset):

    def __init__(self,
                 dataset: TorchDataSource,
                 config: Optional[AwesomeConfig] = None,
                 xytransform: Literal['xy',
                                      'distance_scribble', 'gauss_bubbles'] = 'xy',
                 xytype: Literal['xy', 'feat',
                                 'featxy', 'edge', 'edgexy'] = 'xy',
                 spatio_temporal: bool = False,
                 feature_dir: str = "./data/Feat/",
                 edge_dir: str = "./data/edge/",
                 dimension: Literal['2d', '3d'] = '3d',
                 mode: Literal['model_input', 'sample'] = "model_input",
                 model_input_requires_grad: bool = False,
                 supervision_mode: Literal['weakly', 'full'] = 'weakly',
                 target_encoding: Literal['tensor_mask'] = 'tensor_mask',
                 image_channel_format: Literal['rgb', 'bgr'] = 'rgb',
                 do_image_blurring: bool = False,
                 dtype: torch.dtype = torch.float32,
                 subset: Optional[Union[int, List[int], slice]] = None,
                 no_feature_check: bool = False,
                 returns_index: bool = False,
                 **kwargs
                 ):
        """General dataset class for all individual datasets datasets.

        Parameters
        ----------
        dataset : Dataset
            Dataset which should return a dict which can be further processed by the image sample class.
        config : Optional[AwesomeConfig], optional
            Config which is used for training, by default None
        xytransform : Literal['xy', 'distance_scribble', 'gauss_bubbles'], optional
            The kind of xy transform to apply, by default 'xy'
        xytype : Literal['xy', 'feat', 'featxy'], optional
            The kind of xy type / positional encoding to use, by default 'xy'
        spatio_temporal : bool, optional
            Whether to use spatio temporal data, by default False
            If true, the temporal dimension is added to xy and feature encoding.
        feature_dir : str, optional
            The directory of the precalculated features., by default "./data/Feat/"
        edge_dir : str, optional
            The directory of the precalculated edges., by default "./data/edge/"
        dimension : Literal['2d', '3d', '4d'], optional
            Dimension of the dataset, by default "3d"
            Can be either 2d (pixel-wise prediction), 3d (image-wise prediction)
        mode : Literal['model_input', 'sample'], optional
            Whether to return the model input or the sample, by default "model_input"
        model_input_requires_grad : bool, optional
            Whether the model input should require grad, by default False
            This is needed when the gradient w.r.t the model input should be calculated.
        scribble_percentage : float, optional
            Percentage of scribbles / weakly supervised data to use.
            Percentage less than 1 means that additional random pixels are drawn from the image.
            And all scribbles would then be scribble_percentage * total_pixels.
        dtype : torch.dtype, optional
            The dtype of the data, by default torch.float32
        subset : Optional[Union[int, List[int], slice]], optional
            Subset of the dataset to use, by default None
            Can be used to slice datasets for testing purposes.
        """
        super().__init__(returns_index=returns_index, subset=subset, **kwargs)
        self.__dataset__ = dataset
        self.__images__ = list(range(len(dataset)))
        _ = len(self)  # Enforce creation of subset specifiers
        self.dimension = dimension
        self.model_input_requires_grad = model_input_requires_grad
        self.mode = mode
        self.dtype = dtype
        self.__config__ = config
        self.scribble_percentage = self.__config__.scribble_percentage if self.__config__ is not None else 1.
        self.supervision_mode = supervision_mode
        self.target_encoding = target_encoding
        # Check features
        if "feat" in xytype and not no_feature_check:
            self.check_has_features(
                config, dataset.image_path, feature_dir, xytype)
        if "edge" in xytype:
            if edge_dir is None:
                raise ValueError("Edge dir must be provided if edge is used!")
            if not os.path.exists(edge_dir):
                os.makedirs(edge_dir, exist_ok=True)

        # Prepare data
        for i in tqdm(range(len(self)), "Processing images...", delay=2):
            data_index = self.get_data_index(i)
            self.__images__[data_index] = ImageSample(
                dataset[data_index],
                xytransform=xytransform,
                xytype=xytype,
                mode='scribbles' if (
                    mode == "model_input" and dimension == "2d") else "all",
                feature_dir=feature_dir,
                edge_dir=edge_dir,
                image_channel_format=image_channel_format,
                do_image_blurring=do_image_blurring,
                dtype=dtype,
                spatio_temporal=spatio_temporal,
                t=i,
                t_max=len(self) - 1
            )

    @property
    def do_image_blurring(self) -> bool:
        return self.__images__[self.get_data_index(0)].do_image_blurring

    @do_image_blurring.setter
    def do_image_blurring(self, value: bool) -> float:
        for i in range(len(self)):
            self.__images__[self.get_data_index(i)].do_image_blurring = value

    @property
    def image_channel_format(self) -> Literal['rgb', 'bgr']:
        return self.__images__[self.get_data_index(0)].image_channel_format

    @image_channel_format.setter
    def image_channel_format(self, value: Literal['rgb', 'bgr']) -> float:
        for i in range(len(self)):
            self.__images__[self.get_data_index(
                i)].image_channel_format = value

    def check_has_features(self,
                           config: AwesomeConfig,
                           image_dir: str,
                           feature_dir: str,
                           xytype: str):
        from awesome.run.semantic_soft_segmentation_extractor import SemanticSoftSegmentationExtractor
        if "feat" not in xytype:
            return
        # Check if features are available and items are in the feature directory
        if (not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) != len(self.__dataset__)):
            os.makedirs(feature_dir, exist_ok=True)
            # Create features
            logging.info("Creating semantic features, this takes some time.")
            sss_extrac = SemanticSoftSegmentationExtractor(
                siggraph_sss_code_dir=config.semantic_soft_segmentation_code_dir,
                model_checkpoint_dir=config.semantic_soft_segmentation_model_checkpoint_dir,
                image_dir=image_dir,
                output_dir=feature_dir,
                cpu_only=not config.tf_use_gpu,
                force_creation=False  # Keep already created features
            )
            sss_extrac()

    def split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        train, val = self.__dataset__.split_indices()
        return self._subset_split_indices(train, val)

    def create_subset_mapping(self) -> Dict[int, int]:
        all_images = [(i, v) for i, v in enumerate(self.__images__)]
        if isinstance(self.__subset_specifier__, (list, tuple)):
            images = [all_images[i] for i in self.__subset_specifier__]
        else:
            images = all_images[self.__subset_specifier__]
        if isinstance(images, tuple):
            return {0: images[0]}
        return {i: v[0] for i, v in enumerate(images)}

    def get_dimension_based_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        ret = dict()

        img: torch.Tensor = None
        """Image data."""
        feature_encoding: torch.Tensor = None
        """XY coordinates / Spatial encoding / features"""
        xy: torch.Tensor = None
        """XY coordinates, which are really the xy domain and not encoded."""
        weak_label: torch.Tensor = None
        """Scribble data mask / weak labels if provided."""
        label: torch.Tensor = None
        """Ground truth data Or mask if provided."""

        image_clean: torch.Tensor = sample['clean_image'].to(dtype=self.dtype)
        if self.dimension == "2d":
            img2d = sample['rgb']
            xy2d = sample['xy']
            xy2d_clean = sample['xy_clean']
            scribble2d = sample['scribble'].squeeze(1)
            gt2d = sample['gt'].flatten()

            scribbled_pixels = scribble2d != self.get_number_of_classes()
            img = img2d[scribbled_pixels]
            weak_label = scribble2d[scribbled_pixels]
            label = gt2d[scribbled_pixels]
            feature_encoding = xy2d[scribbled_pixels]
            xy = xy2d_clean[scribbled_pixels]

            if self.scribble_percentage < 1.:
                _rgb, _xy, _clean = self.get_random_pix(
                    img2d, xy2d, xy2d_clean, weak_label.shape[0])
                img = torch.cat([img, _rgb], dim=-2)
                feature_encoding = torch.cat([feature_encoding, _xy], dim=-2)
                xy = torch.cat([xy, _clean], dim=-2)

        elif self.dimension == "3d":
            img = sample['rgb']
            feature_encoding = sample['xy']
            xy = sample['xy_clean']
            weak_label = sample['scribble']
            label = sample['gt']

        else:
            raise ValueError(f"Dimension {self.dimension} not supported!")

        return dict(image=img,
                    feature_encoding=feature_encoding,
                    xy=xy,
                    weak_labels=weak_label,
                    labels=label,
                    clean_image=image_clean)

    def get_ground_truth_indices(self) -> List[int]:
        """Returns the indices of images in the dataset which have ground truth data.

        Returns
        -------
        List[int]
            Indices of the samples with ground truth.
        """
        indices = None
        if hasattr(self.__dataset__, "get_ground_truth_indices"):
            indices = self.__dataset__.get_ground_truth_indices()
            # Convert to subset indices
            indices = [self.get_subset_index(i) for i in indices]
            # Purge None values
            indices = [i for i in indices if i is not None]
        else:
            # Assume all have ground truth
            indices = list(range(len(self)))
        return indices

    @prior()
    def __getitem__(self, index: int):
        sample = self.__images__[self.get_data_index(
            index)].__getitem__(self.dimension)
        if self.mode == "sample":
            return sample
        # For model input prepare data based on dimension

        ret = self.get_dimension_based_data(sample)
        img = ret['image']
        feature_encoding = ret['feature_encoding']
        xy_clean = ret['xy']
        weak_labels = ret['weak_labels']
        clean_image = ret['clean_image']
        ground_truth = ret['labels']

        if self.model_input_requires_grad:
            # Note: We need to detach the tensors else otherwise the gradients might stay accross epochs
            ft_dt = feature_encoding.detach()
            img_dt = img.detach()

            ft_dt.requires_grad = True
            img_dt.requires_grad = True

            feature_encoding = ft_dt
            img = img_dt

        if self.supervision_mode == "weakly":
            target = self.encode_target(weak_labels, index, sample=sample)
        elif self.supervision_mode == "full":
            target = self.encode_target(ground_truth, index, sample=sample)
        else:
            raise ValueError(
                f"Supervision mode {self.supervision_mode} not supported!")

        if not self.returns_index:
            return (img, feature_encoding, xy_clean, dict(clean_image=clean_image)), target
        else:
            return (img, feature_encoding, xy_clean, dict(clean_image=clean_image)), target, torch.tensor(index, dtype=torch.int64)

    def get_random_pix(self, image, xy, xy_clean, interesting_pixels: int) -> torch.Tensor:
        torch.manual_seed(self.split_seed)
        random_pixels = (interesting_pixels *
                         (1 / self.scribble_percentage)) - interesting_pixels
        random_pixels = int(math.ceil(random_pixels))

        needed_pixels = random_pixels
        mask = torch.zeros_like(image[..., 0], dtype=torch.bool)
        indices = torch.argwhere(mask)

        # TODO CHANGE HERE: Provide GT mask and draw random pixels somewhat equally from foreground and background

        max_iter = 1000
        i = 0
        # Looping to be sure to draw enough, we need a loop as randint could draw the same pixel multiple times
        while needed_pixels > 0 and (i < max_iter):
            random_indices_xy = np.random.randint(
                0, image.shape[-2], size=needed_pixels)
            mask[random_indices_xy] = True
            indices = torch.argwhere(mask)
            needed_pixels = random_pixels - indices.shape[0]
            i += 1

        if needed_pixels > 0:
            raise ValueError(
                f"Could not draw enough random pixels after {max_iter} iterations.")

        random_rgb = image[mask]
        random_xy = xy[mask]
        random_xy_clean = xy_clean[mask]
        return random_rgb, random_xy, random_xy_clean

    def __len__(self):
        sub = self.subset_len()
        if sub == NOSUBSET:
            return len(self.__images__)
        return sub

    def encode_target(self, target: torch.Tensor, index: int, sample: Optional[ImageSample] = None) -> torch.Tensor:
        if self.target_encoding == "tensor_mask":
            if self.__config__.use_binary_classification if self.__config__ is not None else True:
                if self.dimension == "2d":
                    # In 2d case, the channel is appended
                    return target.reshape(target.shape + (1,)).to(dtype=self.dtype)
                else:
                    # Add channel if channels are 2 else do nothing
                    if len(target.shape) == 2:
                        target = target.reshape((1,) + target.shape)
                    return target.to(dtype=self.dtype)
            else:
                raise NotImplementedError(
                    "Only binary classification is supported yet!")
        else:
            raise ValueError(
                f"Target encoding {self.target_encoding} not supported!")

    def decode_classes(self, classes: torch.Tensor) -> List[str]:
        """Decodes the classes to the original class labels.

        Parameters
        ----------
        classes : torch.Tensor
            The classes.

        Returns
        -------
        torch.Tensor
            The decoded classes.
        """
        if self.target_encoding == "tensor_mask":
            return classes  # Unknown
        else:
            raise ValueError(
                f"Target encoding {self.target_encoding} not supported!")

    def get_xy_dimension(self) -> int:
        """Returns the feature / channel dimension.

        Returns
        -------
        int
            Feature / channel dimension.
        """
        data_index = self.get_data_index(0)
        return self.__images__[data_index].get_xy_dimension()

    def get_number_of_classes(self) -> int:
        """Returns the number of classes.

        Returns
        -------
        int
            Number of classes.
        """
        data_index = self.get_data_index(0)
        return self.__images__[data_index].get_number_of_classes()

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
        if self.__config__ is None or self.__config__.use_binary_classification:
            if self.dimension == "2d":
                return (output.squeeze(-1) >= 0.5).to(dtype=self.dtype)
            else:
                return (output.squeeze(0) >= 0.5).to(dtype=self.dtype)
        else:
            return output.argmax(dim=-1)

    def get_image(self, index: int) -> torch.Tensor:
        """Returns only the image of the given index.

        Parameters
        ----------
        index : int
            Index of the dataloader where the image should be returned.

        Returns
        -------
        torch.Tensor
            The image.
        """
        with TemporaryProperty(self, dimension="3d", mode="sample", return_prior=False):
            return self[index]['clean_image']
