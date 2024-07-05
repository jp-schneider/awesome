import copy
from dataclasses import dataclass
import logging

import os
import re

import sys
from typing import Any, Dict, List, Optional
import torch
import tensorflow as tfv2
import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
from PIL import Image

"""
This code is  modified from the implementation of https://github.com/iyah4888/SIGGRAPH18SSS

Information about the original and unmodified code:
	@author: Tae-Hyun Oh (http://taehyunoh.com, taehyun@csail.mit.edu)
	@date: Jul 29, 2018
	@description: This is a part of the semantic feature extraction implementation used in 
	[Semantic Soft Segmentation (Aksoy et al., 2018)] (project page: http://people.inf.ethz.ch/aksoyy/sss/).
	This code is modified from the implementation by DrSleep (https://github.com/DrSleep/tensorflow-deeplab-resnet)
	This code is for protyping research ideas; thus, please use this code only for non-commercial purpose only.  
"""

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
"""Original mean values, should be altered for other datasets"""

@dataclass
class HyperColumnDeeplabv2Args():
    batch_size: int = 1
    is_training: bool = False
    num_classes: int = 183

def load_image(image_path: str) -> torch.Tensor:
    img_pil = Image.open(image_path)
    img = np.array(img_pil)
    if len(img.shape) == 3:
        img = img[:,:,0:3]
    else:
        img = img[:,:,None]
    return img


def read_img(image_path, img_mean): 
     

	img_contents = tf.read_file(image_path)
	
	img = tf.image.decode_png(img_contents, channels=3)
	img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
	img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32) # Color Channels are switched?? 
	# Extract mean.
	img -= img_mean

	return img


class SemanticSoftSegmentationExtractor():
	
    IMAGE_PATTERN = r"(?P<image_name>[\d\w]+)\.(?P<extension>png|jpg|jpeg|bmp|tiff|gif|ppm|PNG|JPG|JPEG|BMP|TIFF|GIF|PPM)"
    FEAT_PATTERN = r"(?P<feat_name>[\d\w]+)\.(?P<extension>pt|pth|PT|PTH)"

    def __init__(self, 
				 siggraph_sss_code_dir: str, 
				 model_checkpoint_dir: str,
				 image_dir: Optional[str],
				 output_dir: Optional[str],
                 img_mean: Optional[np.ndarray] = None,
                 cpu_only: bool = True,
                 force_creation: bool = True
                 ) -> None:
        self.siggraph_sss_code_dir = siggraph_sss_code_dir
        self.model_checkpoint_dir = model_checkpoint_dir
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.cpu_only = cpu_only
        self.force_creation = force_creation

        self.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) if img_mean is None else img_mean
        """Original mean values, should be altered for other datasets"""

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
        
        if self.image_dir is not None:
            if not os.path.exists(self.image_dir):
                raise ValueError(f"Image directory does not exist: {self.image_dir}")
        
        if not os.path.exists(self.model_checkpoint_dir):
            raise ValueError(f"Model checkpoint directory does not exist: {self.model_checkpoint_dir}")
        if not os.path.exists(self.siggraph_sss_code_dir):
            raise ValueError(f"Siggraph sss code directory does not exist: {self.siggraph_sss_code_dir}")
		
    def __call__(self, 
                 image_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                **kwargs: Any) -> List[str]:
        # Alter sys
        try:
            sys.path.append(self.siggraph_sss_code_dir)
            tf.reset_default_graph() 
            from deeplab_resnet import HyperColumn_Deeplabv2

            res_path = []
            image_paths = []
            # Calculate image mean


            if image_path is None:
                image_paths = self.read_directory(self.image_dir, self.IMAGE_PATTERN, keys=["image_name", "extension"], primary_key="image_name")
                running_mean = np.zeros((3,))
                running_std = np.zeros((3,))

                _missing_images = None

                # If not force, just create missing paths
                if not self.force_creation:
                    existing_features = self.read_directory(self.output_dir, self.FEAT_PATTERN, keys=["feat_name", "extension"], primary_key="feat_name")
                    existing_features = set(existing_features.keys())
                    _missing_images = {k:path_dict for k, path_dict in image_paths.items() if k not in existing_features}
                    if len(_missing_images) == 0:
                        logging.info("All features already exist, skipping.")
                        return

                for path_dict in tqdm(image_paths.values(), total=len(image_paths), desc="Calculating image mean"):
                    img = load_image(path_dict["path"])
                    running_mean += (1 / len(image_paths)) * np.mean(img, axis=(0,1))
                    running_std += (1 / len(image_paths)) * np.std(img, axis=(0,1))
                
                img_mean_bgr = running_mean[::-1] # Switch to BGR
                output_dir = self.output_dir

                if _missing_images is not None:
                    image_paths = _missing_images
            else:
                img_mean_bgr = self.img_mean
                if output_dir is None:
                    output_dir = os.path.dirname(image_path)
                image_paths = {os.path.split(os.path.splitext(_image_path)[0])[1]: dict(path=image_path, extension=os.path.splitext(image_path)[1])}
   

            args = HyperColumnDeeplabv2Args()
            # Set up tf session and initialize variables. 
            device = None
            #if self.cpu_only:
            #device = tfv2.device("/CPU:0") if self.cpu_only else next(iter(tf.config.list_logical_devices('GPU')), None)
            cp_args = dict()
            device = None
            if self.cpu_only:
                cp_args["device_count"] = {'GPU': 0}
                device = tfv2.device("/CPU:0")
            else:
                cp_args["device_count"] = {'GPU': 1}
                _log_device = next(iter(tf.config.list_logical_devices('GPU')), None)
                if _log_device is None:
                    logging.warning("No GPU found, using CPU instead.")
                    cp_args["device_count"] = {'GPU': 0}
                    device = tfv2.device("/CPU:0")
                else:
                    device = tfv2.device(_log_device.name)
            config = tf.compat.v1.ConfigProto(**cp_args)
            config.gpu_options.allow_growth = True

            tfv2.debugging.set_log_device_placement(True)

            with tf.compat.v1.Session(config=config) as sess, device:
                model = HyperColumn_Deeplabv2(sess, args)

                # Load variables if the checkpoint is provided.
                model.load(self.model_checkpoint_dir)

                for i, (key_val) in tqdm(enumerate(image_paths.items()), total=len(image_paths), desc="Extracting features"):
                    image_name, image_path_dict = key_val
                    if image_path_dict['extension'] == '':
                        continue

                    pad_size = 50
                    _image_path = image_path_dict['path']
                    loaded_image = read_img(_image_path, img_mean = img_mean_bgr)
                    padded_image = tf.pad(loaded_image, [[pad_size,pad_size], [pad_size,pad_size], [0,0]], mode='REFLECT')
                    
                    cur_embed = model.test(padded_image.eval())
                    
                    cur_embed = np.squeeze(cur_embed)
                    img_name = os.path.split(os.path.splitext(_image_path)[0])[1]
                    feat_path = os.path.join(self.output_dir, img_name + '.pt')
                    res = {'embedmap': cur_embed[pad_size:(cur_embed.shape[0]-pad_size),pad_size:(cur_embed.shape[1]-pad_size),:]}
                    torch.save(res, feat_path)
                    res_path.append(feat_path)
        finally:
            sys.path.remove(self.siggraph_sss_code_dir)
        return res_path

    def read_directory(self, 
                       path: str, 
                       pattern: str, 
                       keys: Optional[List[str]] = None,
                       primary_key: str = "name",
                       ) -> Dict[str, Dict[str, Any]]:
        if keys is None:
            keys = ["name"]
        key_data = {key: None for key in keys}
        res = dict()
        for file in os.listdir(path):
            match = re.fullmatch(pattern=pattern, string=file)
            if match:
                data = copy.deepcopy(key_data)
                for k in data.keys():
                    try:
                        v = match.group(k)
                        data[k] = v
                    except IndexError as err:
                        pass
                entry = dict(path=os.path.normpath(os.path.join(path, file)), **data)
                res[entry[primary_key]] = entry
        return res

def main(image_name):
	args = get_arguments()

	# Set up tf session and initialize variables. 
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		model = HyperColumn_Deeplabv2(sess, args)

		# Load variables if the checkpoint is provided.
		model.load(paths.get_path()["model_dir"])

		image_dir = os.path.join(paths.get_path()["root_voc_dir"], 'JPEGImages', image_name + '.jpg')
		
		local_imgflist = [image_dir]
		self.output_dir = paths.get_path()["feature_dir"]
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)

		for i in range(len(local_imgflist)):
			if os.path.splitext(local_imgflist[i])[1] == '':
				continue

			print('{} Processing {}'.format(i, local_imgflist[i]))
			padsize = 50
			
			_, ori_img = read_img(local_imgflist[i], input_size = None, img_mean = IMG_MEAN)
			pad_img = tf.pad(ori_img, [[padsize,padsize], [padsize,padsize], [0,0]], mode='REFLECT')
			cur_embed = model.test(pad_img.eval())
			cur_embed = np.squeeze(cur_embed)
			curfname = os.path.split(os.path.splitext(local_imgflist[i])[0])[1]
			
			self.output_dir = paths.get_path()["feature_dir"]
			cur_svpath = os.path.join(self.output_dir, curfname + '.pt')
			print(cur_svpath)
			res = {'embedmap': cur_embed[padsize:(cur_embed.shape[0]-padsize),padsize:(cur_embed.shape[1]-padsize),:]}
			torch.save(res,cur_svpath)


			