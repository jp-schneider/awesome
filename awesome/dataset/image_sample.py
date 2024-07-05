
import os
from typing import Literal, Optional
import cv2

import torch
from sklearn.decomposition import PCA
from awesome.dataset.data_sample import DataSample

from awesome.dataset.transformator import Transformator
from functools import lru_cache
import numpy as np


class ImageSample():
    
    sample: DataSample

    @property
    def gt(self) -> torch.Tensor:
        """The ground truth label"""
        if self._gt is None:
            gt = self.sample['label']
            if not self.use_memory_cache:
                return gt
            self._gt = gt
        return self._gt

    @property
    def mask(self) -> torch.Tensor:
        """Weak label like scribbles or points"""
        if self._mask is None:
            mask = self.sample['mask']
            if not self.use_memory_cache:
                return mask
            self._mask = mask
        return self._mask
    
    @property
    def image(self) -> torch.Tensor:
        """The image data"""
        if self._image is None:
            image = self.sample['image']
            image = self._process_image(image)
            if not self.use_memory_cache:
                return image
            self._image = image
        return self._image
    
    @property
    def clean_image(self) -> torch.Tensor:
        """Alias for image"""
        if self._clean_image is None:
            clean_image = self.sample['clean_image']
            if not self.use_memory_cache:
                return clean_image
            self._clean_image = clean_image
        return self._clean_image

    @property
    def scribble(self) -> torch.Tensor:
        """Alias for mask"""
        if self._scribble is None:
            scribble = self._get_scribble()
            if not self.use_memory_cache:
                return scribble
            self._scribble = scribble
        return self._scribble

    @property
    def noneclass(self):
        if self._noneclass is None:
            if torch.max(self.gt)==1:
                self._noneclass = torch.tensor(2, dtype=self.dtype)
            else:
                self._noneclass = torch.max(self.gt)
        return self._noneclass

    def _get_scribble(self) -> torch.Tensor:
        if torch.max(self.mask)==2:
            scribble = self.mask
        else:
            scribble = (self.mask * self.gt + (1-self.mask) * self.noneclass)
        return scribble

    @property
    def xy(self) -> torch.Tensor:
        """The xy data"""
        if self._xy is None:
            xy = self._get_xy()
            if not self.use_memory_cache:
                return xy
            self._xy = xy
        return self._xy
    
    @property
    def xy_clean(self) -> torch.Tensor:
        """The xy data"""
        if self._xy_clean is None:
            c, h, w = self.image.shape
            args = dict()
            if self.spatio_temporal:
                args["t"] = self.t
                args["t_max"] = self.t_max
            xy_clean = Transformator.get_positional_matrices(w, h, **args)
            if not self.use_memory_cache:
                return xy_clean
            self._xy_clean = xy_clean
        return self._xy_clean

    def _get_xy(self) -> torch.Tensor:
        if self.xytype == "feat":
            xy = self.feat.squeeze(0)
        elif self.xytype == "featxy":
            xy = Transformator.get_transformation_by_name(self.xytransform, self.scribble, self.xy_clean)
            xy = torch.cat((xy, self.feat.squeeze(0)),dim=0)
        elif self.xytype == "edge":
            xy = self.get_edgemap()
        elif self.xytype == "edgexy":
            _xy = Transformator.get_transformation_by_name(self.xytransform, self.scribble, self.xy_clean)
            edge = self.get_edgemap()
            xy = torch.cat((_xy, edge),dim=0)
        elif self.xytype == "xy":
            xy = self.xy_clean
        else:
            raise ValueError("xytype must be one of 'feat', 'featxy' or 'xy'")
        return xy
    
    @property
    def feat(self) -> torch.Tensor:
        if self._feat is None:
            feat = self.get_semantic_features(self.sample)
            if not self.use_memory_cache:
                return feat
            self._feat = feat
        return self._feat

    def __init__(self, 
                 sample: DataSample,
                 xytransform: Literal['xy', 'distance_scribble', 'gauss_bubbles'] = 'xy',
                 xytype: Literal['xy', 'feat', 'featxy', 'edge'] = 'xy',
                 mode: Literal['all', 'scribbles'] = 'scribbles',
                 feature_dir: str = "./data/Feat/",
                 edge_dir: str = "./data/edge/",
                 do_image_blurring: bool = False,
                 image_channel_format: Literal['rgb', 'bgr'] = 'rgb',
                 use_memory_cache: bool = False,
                 dtype: torch.dtype = torch.float32,
                 spatio_temporal: bool = False,
                 t: Optional[float] = None,
                 t_max: Optional[float] = None,
                 ):
        """Single image representation for all training datasets and classes.
        Individual images are loaded by the AwesomeDataset class whereby this class will wrap single image information.

        Parameters
        ----------
        sample : DataSample
            A datasample from one of the supported datasets.
        xytransform : Literal[&#39;xy&#39;, &#39;distance_scribble&#39;, &#39;gauss_bubbles&#39;], optional
            Which transform on xy coordinates should be used when using it as input, by default 'xy'
        xytype : Literal[&#39;xy&#39;, &#39;feat&#39;, &#39;featxy&#39;], optional
            Mode of channelds to attach additional to the rgb data, by default 'xy'
        mode : Literal[&#39;all&#39;, &#39;scribbles&#39;], optional
            Wether the image should output all pixel / image or just the parts covered by the scribble , by default 'scribbles'
        feature_dir : str, optional
            The directory of the precalculated features., by default "./data/Feat/"
        edge_dir : str, optional
            The directory of the precalculated edge maps., by default "./data/edge/"
        do_image_blurring : bool, optional
            Wether the image should be blurred before training, by default False
        image_channel_format : Literal[&#39;rgb&#39;, &#39;bgr&#39;], optional
            The channel format of the image, by default 'rgb'
        use_memory_cache : bool, optional
            Wether the image should be cached in memory, by default False
        dtype : torch.dtype, optional
            The datatype of the image, by default torch.float32
        spatio_temporal : bool, optional
            Wether the image should be treated as spatio temporal, by default False
        t : Optional[float], optional
            The time of the image, by default None
        t_max : Optional[float], optional
            The maximum time of the image sequence for normalization, by default None
        """
        self.use_memory_cache = use_memory_cache

        self._gt = None
        self._mask = None
        self._image = None
        self._clean_image = None
        self._noneclass = None
        self._feat = None
        self._edge = None
        self._xy = None
        self._xy_clean = None
        self._scribble = None

        self.dtype = dtype
        self.feature_dir = feature_dir
        self.edge_dir = edge_dir

        self.xytransform = xytransform
        self.xytype = xytype
        self.sample = sample
        self.do_image_blurring = do_image_blurring
        self.image_channel_format = image_channel_format
        self.spatio_temporal = spatio_temporal
        self.t = t
        self.t_max = t_max


    def _process_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.do_image_blurring:
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = torch.from_numpy(image).to(dtype=self.dtype).permute(2, 0, 1)
            image = (image / 255)
        if self.image_channel_format == 'bgr':
            image = image[[2, 1, 0], :, :]
        return image

    def get_semantic_features(self, sample):
        # Load Features
        feat_path = os.path.join(self.feature_dir, sample['feat_name']+ '.pt')
        try:
            feat = torch.load(feat_path)
        except Exception as err:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     hypermain(sample['name'])
            # self.feat = torch.load(feat_path)
            raise ValueError("Features not present!") from err

        # Apply PCA to the semantic features
        n_components = 2
        inter_rep = torch.tensor(feat['embedmap'])
        w,h,c = inter_rep.shape
        inter_rep = inter_rep.permute(2,0,1)
        X = inter_rep.reshape([c,-1])
        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Reshape Data
        feat = torch.tensor(pca.components_).reshape([n_components, w, h]).unsqueeze(0).float()
        feat = (feat - torch.min(feat)) / (torch.max(feat) - torch.min(feat))
        return feat

    def get_edgemap(self):
        edge_path = os.path.join(self.edge_dir, self.sample['feat_name']+ '.pth')
        edge = None
        if not os.path.exists(edge_path):
            # Create edge map
            edge = self.create_edge_map(self.clean_image)
            torch.save(edge, edge_path)
        else:
            edge = torch.load(edge_path)
        return edge

    def create_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(1,2,0).numpy()
        image = (image * 255).astype(np.uint8)
        src_ = cv2.GaussianBlur(image, (3, 3), 0)
        gray_ = cv2.cvtColor(src_, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray_, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad_ = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Blur edges
        grad_ = grad_ / 255
        grad_ = cv2.GaussianBlur(grad_, (5, 5), 0)
        edgemap = torch.from_numpy(grad_).to(dtype=self.dtype).unsqueeze(0)
        return edgemap

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
            rgb = self.image.reshape([3,-1]).permute(1,0) # pixel(batchsize) x rgb(3)
            xy = self.xy.reshape([xy_chn,-1]).permute(1,0) # pixel(batchsize) x xy_channels(standard: 2)
            xy_clean = self.xy_clean.reshape([2, -1]).permute(1, 0) # pixel(batchsize) x xy_channels(standard: 2)
            scribble = self.scribble.flatten().unsqueeze(1) # 1 x scribble-label
            gt = self.gt.flatten().unsqueeze(0) # 1 x scribble-label
        if dimensional == "3d":
            rgb = self.image # 3 x w x h
            xy = self.xy # xy_chn x w x h
            xy_clean = self.xy_clean # xy_chn x w x h
            scribble = self.scribble # w x h
            gt = self.gt

        try:
            return {"rgb": rgb.to(dtype=self.dtype),
                    "xy":xy.to(dtype=self.dtype),
                    "xy_clean":xy_clean.to(dtype=self.dtype),
                    "scribble":scribble,
                    "gt": gt,
                    "mask":self.mask,
                    "feat":self.feat.to(dtype=self.dtype) if self.feat is not None else None,
                    "image": self.image.to(dtype=self.dtype), # Original image without changes but maybe with augmentations
                    "clean_image": self.clean_image, # Original image without any changes
                    "raw_sample": self.sample,
                    }
        except Exception as err: 
            return {"rgb": rgb.to(dtype=self.dtype),
                    "xy":xy.to(dtype=self.dtype),
                    "xy_clean":xy_clean.to(dtype=self.dtype),
                    "scribble":scribble,
                    "gt": gt,
                    "mask":self.mask,
                    "feat": rgb.to(dtype=self.dtype),
                    "image": self.image.to(dtype=self.dtype), # Original image without changes but maybe with augmentations
                    "clean_image": self.clean_image, # Original image without any changes
                    "raw_sample": self.sample,
            }
