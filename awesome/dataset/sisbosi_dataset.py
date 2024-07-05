import math
import os
from typing import Any, Dict, List, Literal, Optional, Union
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from PIL import Image
from awesome.dataset.prior_dataset import PriorDataset
from awesome.dataset.subdivisible_dataset import NOSUBSET, SubdivisibleDataset
from awesome.dataset.torch_datasource import TorchDataSource

from awesome.dataset.transformator import Transformator
from awesome.dataset.prior_dataset import PriorDataset, prior
from tqdm.autonotebook import tqdm

from awesome.run.awesome_config import AwesomeConfig
from awesome.util.path_tools import format_os_independent
from awesome.util.temporary_property import TemporaryProperty
from awesome.util.logging import logger

class ScribbleImage():
    def __init__(self,
                 sample,
                 xytransform: Literal['xy',
                                      'distance_scribble', 'gauss_bubbles'] = 'xy',
                 xytype: Literal['xy', 'feat', 'featxy'] = 'xy',
                 mode: Literal['all', 'scribbles'] = 'scribbles',
                 feature_dir: str = "./data/Feat/",
                 dtype: torch.dtype = torch.float32,
                 ):
        """Single image representation for all training datasets and classes.
        Individual images are loaded by the AwesomeDataset class whereby this class will wrap single image information.

        Parameters
        ----------
        sample : torch.Tensor
            The image data.
        xytransform : Literal[&#39;xy&#39;, &#39;distance_scribble&#39;, &#39;gauss_bubbles&#39;], optional
            Which transform on xy coordinates should be used when using it as input, by default 'xy'
        xytype : Literal[&#39;xy&#39;, &#39;feat&#39;, &#39;featxy&#39;], optional
            Mode of channelds to attach additional to the rgb data, by default 'xy'
        mode : Literal[&#39;all&#39;, &#39;scribbles&#39;], optional
            Wether the image should output all pixel / image or just the parts covered by the scribble , by default 'scribbles'
        feature_dir : str, optional
            The directory of the precalculated features., by default "./data/Feat/"
        """
        self.gt = sample["label"]
        self.mask = sample["mask"]
        self.image = sample["image"]
        self.clean_image = torch.tensor(sample["clean_image"]).permute(2, 0, 1)

        self.dtype = dtype
        if torch.max(self.gt) == 1:
            noneclass = torch.tensor(2, dtype=torch.float64)
        else:
            noneclass = torch.max(self.gt)
        if torch.max(self.mask) == 2:
            self.scribble = self.mask
        else:
            self.scribble = (self.mask * self.gt + (1-self.mask) * noneclass)

        self.feature_dir = feature_dir
        feat_path = os.path.join(feature_dir, sample['name'] + '.pt')
        try:
            self.feat = torch.load(feat_path)
        except Exception as err:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     hypermain(sample['name'])
            # self.feat = torch.load(feat_path)
            raise ValueError("Features not present!") from err

        # Image Info
        c, h, w = self.image.shape
        self.nr_pixel = w*h

        # Additional Settings
        self.xytransform = xytransform
        self.xytype = xytype

        # Create Grid
        self.xy = Transformator.get_positional_matrices(w, h)
        self.xy = Transformator.get_transformation_by_name(
            self.xytransform, self.scribble, self.xy)

        self.xy_clean = Transformator.get_positional_matrices(w, h)

        # Get Positions - Training: scribble positions - Validation: all coordinates
        if mode == 'scribbles':
            self.idx_h, self.idx_w = torch.where(self.scribble != noneclass)
        else:
            xxm, yym = torch.meshgrid(torch.arange(
                0, self.image.shape[-2], 1), torch.arange(0, self.image.shape[-1], 1))
            self.idx_h, self.idx_w = xxm.reshape(-1), yym.reshape(-1)

        # Apply PCA to the semantic features
        n_components = 2
        inter_rep = torch.tensor(self.feat['embedmap'])
        w, h, c = inter_rep.shape
        inter_rep = inter_rep.permute(2, 0, 1)
        X = inter_rep.reshape([c, -1])
        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Reshape Data
        feat = torch.tensor(pca.components_).reshape(
            [n_components, w, h]).unsqueeze(0).float()
        self.feat = (feat - torch.min(feat)) / \
            (torch.max(feat) - torch.min(feat))

        if self.xytype == "feat":
            self.xy = feat.squeeze(0)
        elif self.xytype == "featxy":
            self.xy = torch.cat((self.xy, feat.squeeze(0)), dim=0)

    def get_number_of_classes(self):
        '''Returns the number of classes in the current image.'''
        return len(torch.unique(self.gt))

    def get_xy_dimension(self):
        ''' Returns the feature dimension.'''
        return self.xy.shape[0]

    def __getitem__(self, dimensional="3d"):
        ''' Loads the image and its features.

        Args:
            dimensional:
                Can be "2d" or "3d" and decides if the output image has the shape (pixels x 3) for "2d"
                or (1 x 3 x width x height) for "3d".

        Returns:
            A dict including image data of one image and its ground truth and its mask, the features.
        '''

        xy_chn = self.xy.shape[0]
        if dimensional == "2d":
            # pixel(batchsize) x rgb(3)
            rgb = self.image.reshape([3, -1]).permute(1, 0)
            # pixel(batchsize) x xy_channels(standard: 2)
            xy = self.xy.reshape([xy_chn, -1]).permute(1, 0)
            # pixel(batchsize) x xy_channels(standard: 2)
            xy_clean = self.xy_clean.reshape([2, -1]).permute(1, 0)
            scribble = self.scribble.flatten().unsqueeze(1)  # 1 x scribble-label
            gt = self.gt.flatten().unsqueeze(0)  # 1 x scribble-label
        if dimensional == "3d":
            rgb = self.image  # 3 x w x h
            xy = self.xy  # xy_chn x w x h
            xy_clean = self.xy_clean  # xy_chn x w x h
            scribble = self.scribble  # w x h
            gt = self.gt

        try:
            return {"rgb": rgb.to(dtype=self.dtype),
                    "xy": xy.to(dtype=self.dtype),
                    "xy_clean": xy_clean.to(dtype=self.dtype),
                    "scribble": scribble,
                    "gt": gt,
                    "mask": self.mask,
                    "feat": self.feat.to(dtype=self.dtype),
                    # Original image without changes but maybe with augmentations
                    "image": self.image.to(dtype=self.dtype),
                    "clean_image": self.clean_image,  # Original image without any changes
                    }
        except Exception as err:
            return {"rgb": rgb.to(dtype=self.dtype),
                    "xy": xy.to(dtype=self.dtype),
                    "xy_clean": xy_clean.to(dtype=self.dtype),
                    "scribble": scribble,
                    "gt": gt,
                    "mask": self.mask,
                    "feat": rgb.to(dtype=self.dtype),
                    # Original image without changes but maybe with augmentations
                    "image": self.image.to(dtype=self.dtype),
                    "clean_image": self.clean_image,  # Original image without any changes
                    }


class ConvexityDataset(Dataset):
    ''' https://vision.cs.uwaterloo.ca/data/

    '''

    def __init__(self,
                 dataset_path: str = None,
                 transform=True,
                 semantic=False,
                 decoding: bool = False,
                 **kwargs
                 ):
        ''' Initialize.

        Args:
            transform:
                A boolean, indicating if applying data augmentation to the data.
            semantic:
                A boolean indicating if the ground truth is dependend on the semantic meaning of the object.
        '''
        if decoding:
            return
        if dataset_path is None:
            raise ValueError("Dataset path is None!")
        self.semantic = semantic
        # Directories
        root_dir = dataset_path
        # name_dir  = os.path.join(root_dir, 'ImageSets', 'Segmentation')
        self.mask_dir = format_os_independent(
            os.path.join(root_dir, "user_scribbles"))
        self.img_dir = format_os_independent(os.path.join(root_dir, 'img'))
        self.gt_dir = format_os_independent(
            os.path.join(root_dir, 'ground_truth'))

        # Load Names
        # # file = open(os.path.join(name_dir,"{}.txt".format(mode)), 'r')
        # self.img_names = file.read().splitlines()
        # file.close()

        self.patch_size = 300
        self.transform = transform
        self.dataset_len = 51

    def __len__(self):
        '''
        Returns:
            The number of images in the dataset.
        '''

        return self.dataset_len

    def load_ground_truth(self, gt_path):
        img_pil = Image.open(gt_path)
        img = np.array(img_pil)/255.0
        img = np.where(img == 1, 1., 0.)
        return img

    def load_image(self, image_path):
        img_pil = Image.open(image_path)
        img = np.array(img_pil)/255.0
        img = img[:, :, 0:3]
        return img

    def get_masks(self, image_path):
        img_pil = Image.open(image_path)
        img = np.array(img_pil)/255.0
        # Assuming order of: No segmentation = 0, Background = 1, Foreground = 2
        no_mask, bg, fg = np.unique(img)
        bg = np.where(img == bg, 1., 0.)
        fg = np.where(img == fg, 1., 0.)
        return bg, fg

    @staticmethod
    def get_important_pixels(image: torch.Tensor, foreground_mask: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        if foreground_mask is not None and background_mask is not None:
            interesting_pixels = torch.logical_or(
                foreground_mask, background_mask)
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

    def __getitem__(self, idx):
        # Load Image and Ground Truth

        image = self.load_image(self.img_dir + '/' +
                                'img_'+str(idx + 1) + '.png')
        ground_truth = self.load_ground_truth(
            self.gt_dir + '/' + 'GT_'+str(idx + 1) + '.png')

        # Inverting background in ground truth
        # To be 1 for background and 0 for foreground
        ground_truth = 1 - ground_truth

        # extract foreground and background masks
        background_mask, foreground_mask = self.get_masks(
            self.mask_dir + '/' + 'scribbleMask_'+str(idx + 1) + '.png')
        mask = np.full_like(foreground_mask, 2)

        # TODO Background / Foreground encoding switched to be the same as in normal dataset.
        mask[background_mask == 1] = 1
        mask[foreground_mask == 1] = 0
        mask[~foreground_mask.astype(bool) & ~background_mask.astype(bool)] = 2

        clean_image = image
        # Set Augmentation
        if self.transform:
            image, ground_truth, mask = self.data_augmentation(
                image, ground_truth, mask)
            ground_truth = torch.tensor(np.array(ground_truth))
            mask = torch.tensor(np.array(mask))
        else:
            trans = torchvision.transforms.ToTensor()
            image = trans(image)
            ground_truth = torch.tensor(np.array(ground_truth))
            mask = torch.tensor(mask)

        # ground_truth[ground_truth == 1] = int(1)

        if self.semantic == False:
            ground_truth = self.remove_semantic_information(ground_truth)

        sample = {'image': image,
                  'label': ground_truth,
                  'mask': mask,
                  'name': 'img_'+str(idx + 1),
                  'clean_image': clean_image
                  }
        return sample

    def data_augmentation(self, img, gt, mask_b):
        mask = Image.fromarray(np.uint8(mask_b))
        img = Image.fromarray(np.uint8(img * 255))
        gt = Image.fromarray(np.uint8(gt * 255))

        # Random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            gt = TF.hflip(gt)

        # Random Rotation
        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
            gt = TF.rotate(gt, angle)

        trans = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            torchvision.transforms.ToTensor(),
        ])

        img = trans(img)

        # Add Noise
        img = img + torch.randn_like(img) * 0.05
        img = torch.clamp(img, min=0, max=1)

        return img, gt, mask

    def remove_semantic_information(self, image):
        """ Removes the semantic information of the input
        (the input should be thr ground truth data).
        """
        vals = torch.unique(image)
        for i in range(len(vals)):
            image[image == vals[i]] = i
            # print("changed class %.3d --> %.2d"%(vals[i].item(),i))
        return image


class SISBOSIDataset(SubdivisibleDataset, PriorDataset):

    def __init__(self,
                 dataset: Dataset,
                 config: Optional[AwesomeConfig] = None,
                 xytransform: Literal['xy',
                                      'distance_scribble', 'gauss_bubbles'] = 'xy',
                 xytype: Literal['xy', 'feat', 'featxy'] = 'xy',
                 feature_dir: str = "./data/Feat/",
                 bs: Optional[int] = None,
                 dimension: Literal['2d', "3d"] = "3d",
                 mode: Literal['model_input', 'sample'] = "model_input",
                 model_input_requires_grad: bool = False,
                 scribble_percentage: float = 1.,
                 dtype: torch.dtype = torch.float32,
                 subset: Optional[Union[int, List[int], slice]] = None,
                 no_feature_check: bool = False,
                 **kwargs
                 ):
        super().__init__(returns_index=False, subset=subset, **kwargs)
        self.__dataset__ = dataset
        self.__images__ = list(range(len(dataset)))
        _ = len(self)  # Enforce creation of subset specifiers

        if "feat" in xytype and not no_feature_check:
            self.check_has_features(
                config, dataset.img_dir, feature_dir, xytype)

        # Prepare data
        for i in tqdm(range(len(self)), "Processing images..."):
            data_index = self.get_data_index(i)
            self.__images__[data_index] = ScribbleImage(
                dataset[data_index],
                xytransform=xytransform,
                xytype=xytype,
                mode='scribbles' if (
                    mode == "model_input" and dimension == "2d") else "all",
                feature_dir=feature_dir, dtype=dtype)
        self.dimension = dimension
        self.model_input_requires_grad = model_input_requires_grad
        self.mode = mode
        self.bs = bs
        self.dtype = dtype
        self.__config__ = config
        self.scribble_percentage = scribble_percentage

    def check_has_features(self,
                           config: AwesomeConfig,
                           image_dir: str,
                           feature_dir: str,
                           xytype: str):
        if "feat" not in xytype:
            return
        # Check if features are available and items are in the feature directory
        if (not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) != len(self.__dataset__)):
            from awesome.run.semantic_soft_segmentation_extractor import SemanticSoftSegmentationExtractor
            os.makedirs(feature_dir, exist_ok=True)
            # Create features
            logger.info("Creating semantic features, this takes some time.")
            sss_extrac = SemanticSoftSegmentationExtractor(
                siggraph_sss_code_dir=config.semantic_soft_segmentation_code_dir,
                model_checkpoint_dir=config.semantic_soft_segmentation_model_checkpoint_dir,
                image_dir=image_dir,
                output_dir=feature_dir,
                cpu_only=not config.tf_use_gpu,
                force_creation=False  # Keep already created features
            )
            sss_extrac()

    def create_subset_mapping(self) -> Dict[int, int]:
        all_images = [(i, v) for i, v in enumerate(self.__images__)]
        if isinstance(self.__subset_specifier__, (list, tuple)):
            images = [all_images[i] for i in self.__subset_specifier__]
        else:
            images = all_images[self.__subset_specifier__]
        if isinstance(images, tuple):
            return {0: images[0]}
        return {i: v[0] for i, v in enumerate(images)}

    @prior()
    def __getitem__(self, index: int):
        sample = self.__images__[self.get_data_index(
            index)].__getitem__(self.dimension)
        if self.mode == "sample":
            return sample
        # For model input prepare data based on dimension

        image_clean = sample['clean_image'].to(dtype=self.dtype)
        if self.dimension == "2d":
            img2d = sample['rgb']
            xy2d = sample['xy']
            xy2d_clean = sample['xy_clean']
            scribble2d = sample['scribble'].squeeze(1)
            gt2d = sample['gt'].flatten()

            scribbled_pixels = scribble2d != self.get_number_of_classes()
            img = img2d[scribbled_pixels]
            scribble = scribble2d[scribbled_pixels]
            gt = gt2d[scribbled_pixels]
            xy = xy2d[scribbled_pixels]
            xy_clean = xy2d_clean[scribbled_pixels]

            # Select randomly setting["bs"] many pixels
            # if self.bs is not None:
            #     no = min(self.bs, len(gt)-1)
            #     indices = torch.randint(0, len(gt)-1, (no,))
            #     img = img[indices]
            #     scribble = scribble[indices]
            #     gt = gt[indices]
            #     xy = xy[indices]
            #     xy_clean = xy_clean[indices]
            if self.scribble_percentage < 1.:
                _rgb, _xy, _clean = self.get_random_pix(
                    img2d, xy2d, xy2d_clean, scribble.shape[0])
                img = torch.cat([img, _rgb], dim=-2)
                xy = torch.cat([xy, _xy], dim=-2)
                xy_clean = torch.cat([xy_clean, _clean], dim=-2)

            target = self.encode_target(scribble)

            return (img, xy, xy_clean, dict(clean_image=image_clean)), target
        elif self.dimension == "3d":
            img = sample['rgb']
            xy = sample['xy']
            xy_clean = sample['xy_clean']
            scribble = sample['scribble']
            gt = sample['gt']
            if self.model_input_requires_grad:
                xy.requires_grad = True
                xy_clean.requires_grad = True
                img.requires_grad = True
            target = self.encode_target(scribble)

            return (img, xy, xy_clean, dict(clean_image=image_clean)), target
        else:
            raise ValueError(f"Dimension {self.dimension} not supported!")

    def get_random_pix(self, image, xy, xy_clean, interesting_pixels: int) -> torch.Tensor:
        torch.manual_seed(self.split_seed)
        random_pixels = (interesting_pixels *
                         (1 / self.scribble_percentage)) - interesting_pixels
        random_pixels = int(math.ceil(random_pixels))

        needed_pixels = random_pixels
        mask = torch.zeros_like(image[..., 0], dtype=torch.bool)
        indices = torch.argwhere(mask)

        max_iter = 1000
        i = 0
        # Looping to be sure to draw enough
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

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.__config__.use_binary_classification:
            if self.dimension == "2d":
                # In 2d case, the channel is appended
                return target.reshape(target.shape + (1,)).to(dtype=self.dtype)
            else:
                return target.reshape((1,) + target.shape).to(dtype=self.dtype)
        else:
            raise NotImplementedError(
                "Only binary classification is supported yet!")

    def get_xy_dimension(self) -> int:
        """Returns the feature / channel dimension.

        Returns
        -------
        int
            Feature / channel dimension.
        """
        return self.__images__[0].get_xy_dimension()

    def get_number_of_classes(self) -> int:
        """Returns the number of classes.

        Returns
        -------
        int
            Number of classes.
        """
        return self.__images__[0].get_number_of_classes()

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
        if self.__config__.use_binary_classification:
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
