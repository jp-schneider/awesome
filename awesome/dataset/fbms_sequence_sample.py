from collections import OrderedDict
from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Union
import cv2
from matplotlib.figure import Figure
import torch
import os
from awesome.dataset.data_sample import DataSample
from PIL import Image
import numpy as np

from awesome.dataset.label_mode import LabelMode
from awesome.run.functions import channel_masks_to_value_mask, plot_dense_image_mask, plot_mask, plot_mask_labels, save_mask, saveable, value_mask_to_channel_masks

from dataclasses import dataclass
import re
from typing import List
import numpy as np
from awesome.util.torch import VEC_TYPE
import h5py


@dataclass(repr=False)
class FBMSSequenceSample(DataSample):

    index: int = None

    sequence_name: str = None

    image_path: str = None

    ground_truth_path: Optional[str] = None

    trajectory_mask_path: Optional[str] = None

    has_label: bool = False

    _image: Optional[torch.Tensor] = None

    _ground_truth_mask: Optional[torch.Tensor] = None

    _ground_truth_object_ids: Optional[torch.Tensor] = None

    _trajectory_mask: Optional[torch.Tensor] = None

    _trajectory_mask_object_ids: Optional[torch.Tensor] = None

    _weak_label: Optional[torch.Tensor] = None

    _flip_probability: Optional[torch.Tensor] = None

    label_mode: LabelMode = LabelMode.SINGLE_OBJECT

    _label: Optional[torch.Tensor] = None

    foreground_weak_label_object_id: Optional[Union[int, List[int]]] = None
    
    background_weak_label_object_id: Optional[int] = None

    processed_weak_label_path: Optional[str] = None
    confidence_file_path: Optional[str] = None

    do_weak_label_preprocessing: bool = False
    """If true, the weak label is preprocessed before returning it."""
    do_uncertainty_label_flip: bool = False
    """If true, uncertaint pixels can be flipped."""

    _ground_truth_object_id_weak_label_id_mapping: Optional[OrderedDict[int, List[int]]] = None
    _weak_label_id_ground_truth_object_id_mapping: Optional[OrderedDict[int, int]] = None

    check_label_correspondence: bool = True
    """If true, the weak label and ground truth label are checked for correspondence. If not, an exception is raised."""

    foreground_gt_object_id: Optional[Union[int, List[int]]] = None

    foreground_gt_eval_object_id: Optional[int] = None
    """Object id of the ground truth object that is used for evaluation. Fixed by the first frame."""

    use_memory_cache: bool = False

    @property
    def is_ground_truth_evaluatable(self) -> bool:
        """Check if we can evaluate the ground truth for this sample.

        This is the case if the sample has a ground truth and the foreground and background object ids are set
        and the foreground and background object ids are not the same.

        Returns
        -------
        bool
            If the ground truth is evaluatable for metrics.
        """
        if not self.has_label:
            return False
        if self.background_weak_label_object_id is None:
            try:
                _ = self.weak_label
            except Exception as e:
                pass
        if not self.has_valid_foreground_and_background:
            return False
        if self.label_mode == LabelMode.SINGLE_OBJECT:
            fg_gt_mask_id = self.weak_label_id_ground_truth_object_id_mapping.get(self.foreground_weak_label_object_id, None)
            bg_gt_mask_id = self.weak_label_id_ground_truth_object_id_mapping.get(self.background_weak_label_object_id, None)
            return (fg_gt_mask_id is not None and bg_gt_mask_id is not None) and (fg_gt_mask_id != bg_gt_mask_id)
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            bg_gt_mask_id = self.weak_label_id_ground_truth_object_id_mapping.get(self.background_weak_label_object_id, None)
            if bg_gt_mask_id is None:
                return False
            hits = 0
            for fg_id in self.foreground_weak_label_object_id:
                fg_gt_mask_id = self.weak_label_id_ground_truth_object_id_mapping.get(fg_id, None)
                if (fg_gt_mask_id is not None) and (fg_gt_mask_id != bg_gt_mask_id):
                    hits += 1
            if hits == 0:
                # We do not require that all fg. indices are valid, but at least one should be valid
                # As in the case of multiple objects, some labels might not be visible in the frame
                #return False
                logging.warning(f"No valid foreground weak label object id for sample {self.name}!")
            return True
        else:
            raise ValueError(f"Label mode {self.label_mode} not implemented!")

    @property
    def has_valid_foreground_and_background(self) -> bool:
        """Checks if the foreground and background weak label object ids are valid.

        Returns
        -------
        bool
            If the foreground and background weak label object ids are valid.
        """
        if self.foreground_weak_label_object_id is None or self.background_weak_label_object_id is None:
            return False
        ids = self.trajectory_mask_object_ids
        if self.label_mode == LabelMode.SINGLE_OBJECT:
            # Check if the foreground and background are not the same and both are in the weak label
            if self.foreground_weak_label_object_id == self.background_weak_label_object_id:
                return False
            if (self.foreground_weak_label_object_id not in ids) or (self.background_weak_label_object_id not in ids):
                return False
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            if any([x == self.background_weak_label_object_id for x in self.foreground_weak_label_object_id]):
                return False
            has_valid_idx = 0
            for id in self.foreground_weak_label_object_id:
                if (id in ids):
                    has_valid_idx += 1
            if has_valid_idx == 0:
                # We do not require that all fg. indices are valid, but at least one should be valid
                return False
            if self.background_weak_label_object_id not in ids:
                return False
        # If weak_label processing, check also if foreground and background are in the processed weak label
        if self.do_weak_label_preprocessing and self.do_uncertainty_label_flip:
            weak_label = self._get_processed_weak_label(self._get_unprocessed_weak_label())
            wk = weak_label.unsqueeze(0) if len(weak_label.shape) == 2 else weak_label
            # Check that in every channel, there is at least one pixel that is foreground or background
            has_fg = (wk == 0).any(dim=(2)).any(dim=(1))
            has_bg = (wk == 1).any(dim=(2)).any(dim=(1))
            return has_fg.all() and has_bg.all()
        else:
            return True

    @property
    def name(self) -> str:
        return str(self.sequence_name) + ':' + str(self.index)

    @name.setter
    def name(self, value: str):
        pass

    @property
    def feat_name(self) -> Optional[str]:
        if self.image_path is None:
            return None
        return os.path.basename(self.image_path).split('.')[0]

    @feat_name.setter
    def feat_name(self, value: Optional[str]):
        pass

    @property
    def image(self) -> torch.Tensor:
        if self._image is None:
            if self.image_path is None:
                return None
            img = self.load_image(self.image_path)
            if not self.use_memory_cache:
                return img
            self._image = img
        return self._image

    @image.setter
    def image(self, value: torch.Tensor):
        self._image = value

    @property
    def ground_truth_mask(self) -> Optional[torch.Tensor]:
        if self._ground_truth_mask is None:
            if self.ground_truth_path is None:
                return None
            mask, self._ground_truth_object_ids = self.load_mask_multi_channel(
                self.ground_truth_path, ignore_value=None, background_value=0)
            if not self.use_memory_cache:
                return mask
            self._ground_truth_mask = mask
        return self._ground_truth_mask

    @property
    def ground_truth_object_ids(self) -> Optional[torch.Tensor]:
        if self._ground_truth_object_ids is None:
            if self.ground_truth_path is None:
                return None
            _, self._ground_truth_object_ids = self.load_mask_multi_channel(
                self.ground_truth_path, ignore_value=None, background_value=0)
        return self._ground_truth_object_ids

    @property
    def trajectory_mask(self) -> Optional[torch.Tensor]:
        if self._trajectory_mask is None:
            if self.trajectory_mask_path is None:
                return None
            mask, self._trajectory_mask_object_ids = self.load_mask_multi_channel(
                self.trajectory_mask_path, ignore_value=None, background_value=0)
            if not self.use_memory_cache:
                return mask
            self._trajectory_mask = mask
        return self._trajectory_mask

    @property
    def ground_truth_object_id_weak_label_mapping(self) -> OrderedDict[int, List[int]]:
        """Gets the mapping from ground truth object ids to weak label object ids.
        As the object could potentially be split into multiple objects in the weak label,
        all of these are returned in the order of largest pixel support (descending), e.g. largest object first.

        The largest gt object is also returned first. This is usually the background, which is artificially added by the 0th class.

        If the frame has no ground truth, the mapping will be the identity mapping (Trajectory mask to self), 
        with an additional unlabeled object id 0 (noneclass) and also in order of their largest pixel support.

        Returns
        -------
        Optional[OrderedDict[int, List[int]]]
            Mapping from ground truth object id to weak label object ids.
        """
        if self._ground_truth_object_id_weak_label_id_mapping is None:
            self._ground_truth_object_id_weak_label_id_mapping = self._get_gt_object_id_weak_label_mapping()
        return self._ground_truth_object_id_weak_label_id_mapping

    @property
    def weak_label_id_ground_truth_object_id_mapping(self) -> OrderedDict[int, Optional[int]]:
        """Gets the mapping from weak label object ids to ground truth object ids.
        As the object could potentially be split into multiple objects in the weak label,
        all of these match the same ground truth object id.

        The largest gt correspondings object is also returned first. This is usually the background, which is artificially added by the 0th class.

        Some weak labels might not have a corresponding ground truth object id, e.g. if the object is not visible in the frame, or the object is not annotated / wrongly classified.

        Returns
        -------
        OrderedDict[int, int]
            Mapping from weak label object id to ground truth object ids.
        """
        if self._weak_label_id_ground_truth_object_id_mapping is None:
            self._weak_label_id_ground_truth_object_id_mapping = {
                v: k for k, y in self.ground_truth_object_id_weak_label_mapping.items() for v in y}
        return self._weak_label_id_ground_truth_object_id_mapping

    @trajectory_mask.setter
    def trajectory_mask(self, value: torch.Tensor):
        self._trajectory_mask = value


    @property
    def weak_label(self):
        if self._weak_label is None:
            mask = self._get_weak_label()
            if not self.use_memory_cache or self.do_weak_label_preprocessing:
                return mask
            self._weak_label = mask
        return self._weak_label

    @weak_label.setter
    def weak_label(self, value: torch.Tensor):
        self._weak_label = value

    def _check_foreground_object_set(self, mask: torch.Tensor = None, oids: torch.Tensor = None):
        if self.label_mode == LabelMode.SINGLE_OBJECT:
            self._check_foreground_object_set_single_label(mask, oids)
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            self._check_foreground_object_set_multi_label(mask, oids)
        else:
            raise ValueError("Unknown label mode for checking foreground object set!")
            
    def _check_foreground_object_set_single_label(self, mask: torch.Tensor = None, oids: torch.Tensor = None):
        if not self.label_mode == LabelMode.SINGLE_OBJECT:
            raise ValueError("This method is only applicable for single object label mode!")
        
        if mask is None:
            mask = self.trajectory_mask
        if oids is None:
            oids = self.trajectory_mask_object_ids

        _support = mask.sum(dim=(1, 2))
        _idx_rank = torch.argsort(_support, descending=True)
        # Get class with most pixels as background of not specified
        # Second most will be the foreground if not specified
        mapping = self.weak_label_id_ground_truth_object_id_mapping

        if self.foreground_weak_label_object_id is None:
            _selected_id = None
            _mapping = self.ground_truth_object_id_weak_label_mapping
            if self.has_label and self.check_label_correspondence:
                _gt_id, matched_ids = next(((k, v) for k, v in _mapping.items() if k != 0 and len(v) > 0), (None, None))
                if _gt_id is None:
                    raise Exception(f"Could not find a foreground weak label object id for sample {self.name}!")
                if matched_ids is None or len(matched_ids) == 0:
                    raise Exception(f"Could not find a foreground weak label object id for sample {self.name}!")
                self.foreground_gt_object_id = _gt_id
                _selected_id = matched_ids[0]
            else:
                _selected_id = oids[_idx_rank[1]].item()
            if isinstance(_selected_id, torch.Tensor):
                _selected_id = _selected_id.item()
            self.foreground_weak_label_object_id = _selected_id
        elif self.has_label and self.check_label_correspondence:
            mapping = self.weak_label_id_ground_truth_object_id_mapping
            gt_id = mapping.get(self.foreground_weak_label_object_id, None)
            if gt_id is None:
                raise Exception(
                    f"Foreground weak label object id {self.foreground_weak_label_object_id} is not mapped to ground truth!")
        if self.background_weak_label_object_id is None:
            _selected_id = None
            _mapping = self.ground_truth_object_id_weak_label_mapping
            if self.has_label and self.check_label_correspondence:
                _gt_id = 0 if ((0 in _mapping) and len(_mapping.get(0)) > 0) else next(k for k in _mapping.keys())  # 0 Is background
                matched_ids = _mapping.get(_gt_id, None)
                if _gt_id is None:
                    raise Exception(f"Could not find a background weak label object id for sample {self.name}!")
                if matched_ids is None or len(matched_ids) == 0:
                    raise Exception(f"Could not find a background weak label object id for sample {self.name}!")
                _selected_id = matched_ids[0]
            else:
                # Take largest if not the foreground id is the largest, otherwise take second largest
                _selected_id = oids[_idx_rank[0]].item() if oids[_idx_rank[0]].item() != self.foreground_weak_label_object_id else oids[_idx_rank[1]].item()
            if isinstance(_selected_id, torch.Tensor):
                _selected_id = _selected_id.item()
            self.background_weak_label_object_id = _selected_id
        elif self.has_label and self.check_label_correspondence:
            # Check that the background and foreground are not the same as the ground truth
            # As well as these have dedicated objects assigned.
            mapping = self.weak_label_id_ground_truth_object_id_mapping
            bg_id = self.background_weak_label_object_id
            if isinstance(bg_id, torch.Tensor):
                bg_id = bg_id.item()
            gt_id = mapping.get(bg_id, None)
            if gt_id is None:
                raise Exception(
                    f"Background weak label object id {self.background_weak_label_object_id} is not mapped to ground truth!")
            elif gt_id == self.foreground_gt_object_id:

                raise Exception(
                     f"Background weak label object id {self.background_weak_label_object_id} is mapped to the same ground truth object id as the foreground!")

    def _check_foreground_object_set_multi_label(self, mask: torch.Tensor = None, oids: torch.Tensor = None):
        if not self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            raise ValueError("This method is only applicable for multi objects label mode!")
        
        if mask is None:
            mask = self.trajectory_mask
        if oids is None:
            oids = self.trajectory_mask_object_ids

        _support = mask.sum(dim=(1, 2))
        _idx_rank = torch.argsort(_support, descending=True)
        
        # Get class with most pixels as background if not specified
        # All other classes will be the foreground if not specified
        mapping = self.weak_label_id_ground_truth_object_id_mapping

        if self.foreground_weak_label_object_id is None:
            self.foreground_weak_label_object_id = oids[_idx_rank[1:]].tolist()
        
        if self.has_label and self.check_label_correspondence:
            gt_ids = []
            hits = 0
            for fg_id in self.foreground_weak_label_object_id:
                gt_id = mapping.get(fg_id, None)
                if gt_id is not None:
                    hits += 1
                gt_ids.append(gt_id)
            if hits == 0:
                raise ValueError(f"No foreground weak label object id for sample {self.name} is mapped to ground truth!")
            if self.foreground_gt_object_id is None:
                self.foreground_gt_object_id = gt_ids
        
        if self.background_weak_label_object_id is None:
            _selected_id = None
            _mapping = self.ground_truth_object_id_weak_label_mapping
            if self.has_label and self.check_label_correspondence:
                _gt_id = 0 if ((0 in _mapping) and len(_mapping.get(0)) > 0) else next(k for k in _mapping.keys())  # 0 Is background
                matched_ids = _mapping.get(_gt_id, None)
                if _gt_id is None:
                    raise ValueError(f"Could not find a background weak label object id for sample {self.name}!")
                if matched_ids is None or len(matched_ids) == 0:
                    raise ValueError(f"Could not find a background weak label object id for sample {self.name}!")
                _selected_id = matched_ids[0]
            else:
                # Take largest if not the foreground id is the largest, otherwise take second largest
                _selected_id = oids[_idx_rank[0]].item() if oids[_idx_rank[0]].item() != self.foreground_weak_label_object_id else oids[_idx_rank[1]].item()
            if isinstance(_selected_id, torch.Tensor):
                _selected_id = _selected_id.item()
            self.background_weak_label_object_id = _selected_id
        if self.has_label and self.check_label_correspondence:
            # Check that the background and foreground are not the same as the ground truth
            # As well as these have dedicated objects assigned.
            mapping = self.weak_label_id_ground_truth_object_id_mapping
            bg_id = self.background_weak_label_object_id
            if isinstance(bg_id, torch.Tensor):
                bg_id = bg_id.item()
            gt_id = mapping.get(bg_id, None)
            if gt_id is None:
                raise ValueError(
                    f"Background weak label object id {self.background_weak_label_object_id} is not mapped to ground truth!")
            elif gt_id == self.foreground_gt_object_id:
                raise ValueError(
                     f"Background weak label object id {self.background_weak_label_object_id} is mapped to the same ground truth object id as the foreground!")

    @property
    def flip_label_probability_path(self) -> Optional[str]:
        if self.processed_weak_label_path is None:
            return None
        return self.processed_weak_label_path.replace(".png", "_flip_probability.npy")


    def _get_unprocessed_weak_label(self) -> torch.Tensor:
        mask = self.trajectory_mask
        oids = self.trajectory_mask_object_ids
        self._check_foreground_object_set(mask=mask, oids=oids)

        if self.label_mode == LabelMode.SINGLE_OBJECT:
            fg_mask = mask[torch.argwhere(oids == self.foreground_weak_label_object_id).item()].bool()
            bg_mask = mask[torch.argwhere(oids == self.background_weak_label_object_id).item()].bool()
            
            weak_label = torch.fill(torch.zeros_like(fg_mask, dtype=torch.float32), self.WEAK_LABEL_NONECLASS)
            weak_label[fg_mask] = self.WEAK_LABEL_FOREGROUND
            weak_label[bg_mask] = self.WEAK_LABEL_BACKGROUND

            return weak_label
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            # Create a channel mask for each object
            # In multi object mode, the weak label is the class index for cross entropy loss
            # The background is the first class, having the lowest index
            self.WEAK_LABEL_NONECLASS = -1
            weak_label = torch.fill(torch.zeros((1, *mask.shape[-2:]), dtype=torch.float32), self.WEAK_LABEL_NONECLASS)
            bg_mask = mask[torch.argwhere(oids == self.background_weak_label_object_id).item()].bool()
            
            self.WEAK_LABEL_BACKGROUND = 0
            weak_label[0, bg_mask] = self.WEAK_LABEL_BACKGROUND

            for i, fg_oid in enumerate(self.foreground_weak_label_object_id):
                fg_mask = mask[torch.argwhere(oids == fg_oid).item()].bool()
                weak_label[0, fg_mask] = i + 1
           
            self.WEAK_LABEL_FOREGROUND = list(range(1, len(self.foreground_weak_label_object_id) + 1))
            return weak_label
        else:
            raise ValueError(f"Label mode {self.label_mode} not implemented!")

    def _get_processed_weak_label(self, weak_label: torch.Tensor) -> Optional[torch.Tensor]:
        if self.do_weak_label_preprocessing:
            # Implemented weak label preprocessing with probability flip as suggested by Kardoost et al.
            if self.label_mode != LabelMode.SINGLE_OBJECT:
                raise NotImplementedError("Weak label preprocessing was implemented only implemented for single object labels in the work of Kardoost et al.!")
            flip = None
            if self.processed_weak_label_path is not None:
                if not os.path.exists(self.processed_weak_label_path) or not os.path.exists(self.flip_label_probability_path):
                    # Create from trajectory mask
                    os.makedirs(os.path.dirname(self.processed_weak_label_path), exist_ok=True)
                    weak_label, flip = self._process_weak_label(weak_label)
                    self._flip_probability = flip
                    # Saving processed weak label and flip probability
                    save_mask(weak_label, self.processed_weak_label_path)
                    np.save(self.flip_label_probability_path, flip.numpy())
                else:
                    weak_label = self.load_mask(self.processed_weak_label_path).squeeze()
                    flip = torch.tensor(np.load(self.flip_label_probability_path))
                    self._flip_probability = flip
            if self.do_uncertainty_label_flip:
                weak_label = self._flip_on_probability(weak_label, flip)
            return weak_label
        else:
            return None

    def _get_weak_label(self) -> torch.Tensor:
        """Gets the waek label for the sample.
        This is based on the label mode.

        Returns
        -------
        torch.Tensor
            The weak label.
        """
        if self.do_weak_label_preprocessing:
            return self._get_processed_weak_label(self._get_unprocessed_weak_label())
        else:
            return self._get_unprocessed_weak_label()

    def _closest_node(self, node, nodes, ii):
        nodes = np.asarray(nodes)
        node = np.expand_dims(node, axis=1)
        dist_2 = np.sum((nodes - node)**2, axis=0)
        dist_2[ii] = 10000.0  # insert large value for node itself (o.w. the value will be 0)
        return np.min(dist_2)

    def _process_weak_label(self, weak_label: torch.Tensor) -> torch.Tensor:
        # This is a slightly optimized version of the original code from the paper
        mask = weak_label.numpy()
        if self.confidence_file_path is None:
            raise ValueError("Confidence file path is not set!")
        confidence = np.expand_dims(np.asarray(h5py.File(self.confidence_file_path, 'r')['confidence']), 0)

        n_classes = 2  # Currently support only binary classification
        OUTLIER_THRESHOLD = 1000 # In pixels

        # Convert mask to format of Self-Supervised ... format
        # 0: background, 1: foreground, -1: none
        _conv_mask = np.zeros_like(mask)
        _conv_mask[...] = -1

        _conv_mask[mask == 0] = 1
        _conv_mask[mask == 1] = 0

        mask = _conv_mask
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, 2)

        pos_fg = np.asarray(np.where(mask[:, :, 0] == 1))
        for ii in range(len(pos_fg[0])):
            distance_ = self._closest_node(pos_fg[:, ii], pos_fg, ii)
            if distance_ > OUTLIER_THRESHOLD:
                mask[pos_fg[0, ii], pos_fg[1, ii], 0] = -1

        mask = np.transpose(mask, axes=[2, 0, 1])
        # Mask in format (C, H, W) where C is the number of classes

        confidence_temp = np.copy(confidence)
        confidence_temp[np.where(confidence_temp == -1)] = 100
        uncertain_points = np.where(confidence_temp < 0.4)  # less values correspond to more uncertain points
        mask_temp = np.copy(mask)

        #_save_debug = True
        #save_dir = "./data/temp/"
        #os.makedirs(save_dir, exist_ok=True)
        
        #if _save_debug:
        #    np.save(os.path.join(save_dir + "before_uncertainty.npy"), mask_temp)

        flip_probabilities = np.zeros((len(uncertain_points[0]), 4))
        for itr1 in range(len(uncertain_points[0])):

            # take patch of size 128*128 around the uncertain point
            row_pos = uncertain_points[1][itr1]
            col_pos = uncertain_points[2][itr1]

            label_of_point = mask[0,row_pos,col_pos]

            if mask[0,row_pos, col_pos] == -1:
                continue

            start_row = np.max([0, row_pos-64])
            end_row = np.min([row_pos+64, confidence.shape[1]])
            start_col = np.max([0, col_pos-64])
            end_col = np.min([col_pos+64, confidence.shape[2]])
            patch = np.copy(mask[0,start_row:end_row, start_col:end_col])
            
            labels_in_patch = np.unique(patch)

            NONE_LABEL = -1
            BACKGROUND_LABEL = 0
            FOREGROUND_LABEL = 1
            
            # _labels_in_patch, counts = np.unique(patch, return_counts=True)
            # _votes_in_patch = counts[np.argwhere(_labels_in_patch != NONE_LABEL)].squeeze()
            # _votes_in_patch = _votes_in_patch / np.sum(_votes_in_patch) # Relative votes per class
             
            # _all_labels_values, _all_labels_counts = np.unique(mask, return_counts=True)
            # order = [np.argwhere(x == _all_labels_values).squeeze() for i, x in enumerate(_labels_in_patch) if x in _all_labels_values and x != NONE_LABEL]
            # _all_labels_counts = _all_labels_counts[order]

            # _all_labels_counts = np.flip(_all_labels_counts)
            # _votes_in_patch = np.flip(_votes_in_patch)
            # _labels_in_patch = np.flip(_labels_in_patch)
            



            #if patch.shape[0]==128 and patch.shape[1]==128 and len(np.unique(patch))>3:
            #    pdb.set_trace()
            
            if len(labels_in_patch) <= 2: # labels will be -1, x
                continue
            else:
                #patch[np.where(patch==mask[0,row_pos,col_pos])] = -1
                votes_in_patch = []
                for itr2 in range(len(labels_in_patch)):
                    votes_in_patch.append(len(np.where(patch==labels_in_patch[itr2])[0]))
                votes_in_patch.pop(0)
                
                all_labels = []
                for itr2 in range(len(labels_in_patch)):
                    all_labels.append(len(np.where(mask == labels_in_patch[itr2])[0]))
                all_labels.pop(0)
                #max_all_labels = max(all_labels)
                #for itr2 in range(len(all_labels)):
                #    all_labels[itr2] = max_all_labels/all_labels[itr2]

                votes_in_patch /= np.sum(votes_in_patch)
                
                for itr2 in range(len(votes_in_patch)):
                    if all_labels[itr2] > 900: # most probably a bg label
                        votes_in_patch[itr2] = np.min([votes_in_patch[itr2], 0.3])

                #votes_in_patch *= all_labels
                #votes_in_patch[np.where(votes_in_patch>=1.0)] = 0.99
                all_labels = np.flip(all_labels)
                votes_in_patch = np.flip(votes_in_patch)
                labels_in_patch = np.flip(labels_in_patch)

                # if the label belong to bg (longest label), do not change it to other labels
                length_of_label_of_point = len(np.where(mask == label_of_point)[0]) # Poor! Better: all_labels[(labels_in_patch == label_of_point)[:2]].item()

                # # TODO I dont like this voting here....
                # if length_of_label_of_point <= 900: 
                #     for itr2 in range(len(votes_in_patch)):
                #         # if the label belong to fg, it is possible to change it to another labels
                #         if np.random.rand() < votes_in_patch[itr2]:
                #             if all_labels[itr2] > 900 and labels_in_patch[itr2] == label_of_point:
                #                 continue
                #             else:
                #                 mask_temp[0,row_pos, col_pos] = labels_in_patch[itr2]
                #                 break
                #         else:
                #             pass
                
                
                # probability as FG, BG
                flip_probabilities[itr1, 0] = row_pos
                flip_probabilities[itr1, 1] = col_pos

                #probability[iter1, 2] = FG FLIP PROBABILITY
                #probability[iter1, 3] = BG FLIP PROBABILITY

                if length_of_label_of_point <= 900: 
                    for itr2 in range(len(votes_in_patch)):
                        if all_labels[itr2] > 900 and labels_in_patch[itr2] == label_of_point:
                            flip_probabilities[itr1, 2 + itr2] = 0
                        else:
                            flip_probabilities[itr1, 2 + itr2] = votes_in_patch[itr2]
                pass


        # pdb.set_trace()
        mask = np.copy(mask_temp)
        mask_temp = mask[0, :, :]


        # TODO TBD: Is 30 an appropriate value? For thinkness of the "border", why changing every second pixel and not in same density as the optical flow predicts?
        # Nonetheless, original paper proposed this...

        BORDER_THICKNESS = 30
        DENSITY = 2
        for itr1 in range(0, mask_temp.shape[0]-BORDER_THICKNESS, 5):
            box_temp = mask_temp[itr1:itr1+BORDER_THICKNESS, 0:BORDER_THICKNESS]
            if len(np.unique(box_temp)) < 2:
                mask[0, itr1:itr1+BORDER_THICKNESS:DENSITY, 0:BORDER_THICKNESS:DENSITY] = 0

        for itr1 in range(0, mask_temp.shape[1]-BORDER_THICKNESS, 5):
            box_temp = mask_temp[0:BORDER_THICKNESS, itr1:itr1+BORDER_THICKNESS]
            if len(np.unique(box_temp)) < 2:
                mask[0, 0:BORDER_THICKNESS:DENSITY, itr1:itr1+BORDER_THICKNESS:DENSITY] = 0

        for itr1 in range(0, mask_temp.shape[0]-BORDER_THICKNESS, 5):
            box_temp = mask_temp[itr1:itr1+BORDER_THICKNESS, -BORDER_THICKNESS:-1]
            if len(np.unique(box_temp)) < 2:
                mask[0, itr1:itr1+BORDER_THICKNESS:DENSITY, -BORDER_THICKNESS:-1:DENSITY] = 0

        for itr1 in range(0, mask_temp.shape[1]-BORDER_THICKNESS, 5):
            box_temp = mask_temp[-BORDER_THICKNESS:-1, itr1:itr1+BORDER_THICKNESS]
            if len(np.unique(box_temp)) < 2:
                mask[0, -BORDER_THICKNESS:-1:DENSITY, itr1:itr1+BORDER_THICKNESS:DENSITY] = 0

        # Convert mask to format of our format
        # 1: background, 0: foreground, 2: none

        _conv_mask = np.zeros_like(mask)
        _conv_mask[...] = 2
        _conv_mask[mask == 0] = 1
        _conv_mask[mask == 1] = 0

        return torch.tensor(_conv_mask).squeeze(), torch.tensor(flip_probabilities)

    def _flip_on_probability(self, mask: torch.Tensor, flip_probability: torch.Tensor) -> torch.Tensor:
        mask = mask.clone()
        flip_fg = flip_probability[:, 2] 
        flip_bg = flip_probability[:, 3] 
        fg_flip_mask = torch.zeros(len(flip_probability), dtype=torch.bool)
        bg_flip_mask = torch.zeros(len(flip_probability), dtype=torch.bool)
        probas = torch.rand((len(flip_probability), 2))

        fg_flip_mask[probas[:, 0] < flip_fg] = True
        bg_flip_mask[probas[:, 1] < flip_bg] = True
        bg_flip_mask[fg_flip_mask] = False

        fg_flip_coords = flip_probability[torch.argwhere(fg_flip_mask).squeeze(), :2].int().T
        bg_flip_coords = flip_probability[torch.argwhere(bg_flip_mask).squeeze(), :2].int().T
      
        mask[fg_flip_coords[0], fg_flip_coords[1]] = 0 # This should change nothing as code prevents it
        mask[bg_flip_coords[0], bg_flip_coords[1]] = 1
        return mask

    def _get_gt_foreground_label_id(self) -> Union[int, List[int]]:
        if self.label_mode == LabelMode.SINGLE_OBJECT:
            if self.foreground_gt_object_id is None:
                mask = self.ground_truth_mask
                oids = self.ground_truth_object_ids
                _ = self.weak_label  # Assure wl object ids are set

                mapped = self.weak_label_id_ground_truth_object_id_mapping.get(
                    self.foreground_weak_label_object_id, None)
                if mapped is None:
                    if self.check_label_correspondence:
                        raise Exception(
                            f"Foreground weak label object id {self.foreground_weak_label_object_id} is not mapped to ground truth!")
                    else:
                        logging.warning(
                            f"Foreground weak label object id {self.foreground_weak_label_object_id} is not mapped to ground truth! Chosing any, this will lead to wrong results!")
                        _support = mask.sum(dim=(1, 2))
                        _idx_rank = torch.argsort(_support, descending=True)
                        self.foreground_gt_object_id = oids[_idx_rank[0]]
                else:
                    self.foreground_gt_object_id = mapped
            return self.foreground_gt_object_id
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            if self.foreground_gt_object_id is None:
                raise ValueError("Foreground ground truth object id is not set and auto assign is not supported for multiple objects!")
            if len(self.foreground_gt_object_id) != len(self.foreground_weak_label_object_id):
                raise ValueError("Foreground ground truth object id and weak label object id do not have the same length!")
            return self.foreground_gt_object_id

    def _get_gt_label(self) -> torch.Tensor:
        mask = self.ground_truth_mask
        oids = self.ground_truth_object_ids

        if self.label_mode == LabelMode.SINGLE_OBJECT:
            fg_mask = torch.zeros_like(mask[0]).bool()
            gt_oid = self._get_gt_foreground_label_id()
            if gt_oid not in oids:
                fg_mask = torch.zeros_like(mask[0]).bool()
                # Display warning => Gt does not contain the wanted object id. Maybe it is not visible in the frame?
                logging.warning(f"Ground truth does not contain the wanted foreground object id {self.foreground_gt_object_id}! Maybe it is not visible in the frame?")
            else:
                fg_mask = mask[torch.argwhere(oids == gt_oid).item()].bool()
            label = torch.ones_like(fg_mask, dtype=torch.float32)
            label[fg_mask] = 0.
            return label
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            gt_oids = self._get_gt_foreground_label_id()
            selected_channels = [torch.argwhere(oids == x).squeeze() for x in gt_oids]
            selected_channels = [x.item() for x in selected_channels if x is not None and x.numel() > 0]
            selected_oids = mask[selected_channels]
            return torch.from_numpy(channel_masks_to_value_mask(selected_oids, object_values=np.arange(1, len(gt_oids) + 1), base_value=0)).unsqueeze(0)
        else:
            raise NotImplementedError()

    def _get_gt_object_id_weak_label_mapping(self, min_threshold: float = 0.5) -> OrderedDict[int, List[int]]:
        """Maps the ground truth object ids to the weak label object ids.
        As the object could potentially be split into multiple objects in the weak label,
        all of these are returned in the order of largest pixel support (descending), e.g. largest object first.

        The largest gt object is also returned first.

        Returns
        -------
        OrderedDict[int, List[int]]
            Mapping from ground truth object id to weak label object ids.
        """
        mask = self.ground_truth_mask  # Should be shape (C, H, W) C is number of masks
        gt_ids = self.ground_truth_object_ids

        if not self.has_label:
            mask = self.trajectory_mask
            gt_ids = self.trajectory_mask_object_ids

            mask = torch.cat([mask, torch.zeros((1,) + mask.shape[1:])], dim=0)
            mask[-1, ...] = torch.logical_not(mask[:-1, ...].any(dim=0))
            gt_ids = torch.cat([gt_ids, torch.tensor([0])], dim=0)
            _support = mask.sum(dim=(1, 2))
            _gt_idx_rank = torch.argsort(_support, descending=True)
            mapping = OrderedDict([(gt_ids[i].item(), [gt_ids[i].item()] if gt_ids[i] != 0 else []) for i in _gt_idx_rank])
            return mapping

        if len(mask.shape) == 4:
            raise ValueError("Mask has more than 3 dimensions, this is not supported!")

        # Add gt background as channel
        mask = torch.cat([mask, torch.zeros((1,) + mask.shape[1:])], dim=0)
        mask[-1, ...] = torch.logical_not(mask[:-1, ...].any(dim=0))
        gt_ids = torch.cat([gt_ids, torch.tensor([0])], dim=0)

        _support = mask.sum(dim=(1, 2))
        _gt_idx_rank = torch.argsort(_support, descending=True)

        weak_masks = self.trajectory_mask
        weak_ids = self.trajectory_mask_object_ids

        mapping = OrderedDict()

        for i, gt_id_idx in enumerate(_gt_idx_rank):
            # Calculate support for each mask
            encounter = torch.logical_and(mask[gt_id_idx], weak_masks)

            eq_support = encounter.sum(dim=(1, 2))
            total_pix = weak_masks.sum(dim=(1, 2))
            match_ratio = eq_support / total_pix

            # Get best match which is the largest object with at least min_threshold overlap
            ordered_matches = torch.argsort(total_pix, descending=True)

            # Get all matches above threshold
            matches = ordered_matches[match_ratio[ordered_matches] >= min_threshold]
            matched_ids = weak_ids[matches]
            mapping[gt_ids[gt_id_idx].item()] = matched_ids.tolist()

        return mapping

    @property
    def trajectory_mask_object_ids(self) -> Optional[torch.Tensor]:
        if self._trajectory_mask_object_ids is None:
            if self.trajectory_mask_path is None:
                return None
            mask, self._trajectory_mask_object_ids = self.load_mask_multi_channel(
                self.trajectory_mask_path, ignore_value=None, background_value=0)
            if self.use_memory_cache:
                self._trajectory_mask = mask
        return self._trajectory_mask_object_ids

    @property
    def label(self) -> torch.Tensor:
        if self._label is None:
            if not self.has_label:
                return None 
            mask = self._get_gt_label()
            if not self.use_memory_cache:
                return mask
            self._label = mask
        return self._label

    @label.setter
    def label(self, value: torch.Tensor):
        self._label = value

    def load_image(self, image_path: str) -> torch.Tensor:
        img_pil = Image.open(image_path)
        img = np.array(img_pil, dtype='float')/255.0
        if len(img.shape) == 3:
            img = img[:, :, 0:3]
        else:
            img = img[:, :, None]
        return torch.from_numpy(img.transpose(2, 0, 1))

    def load_mask(self,
                  image_path: str,
                  has_contour: bool = False,
                  background_value: int = 0
                  ) -> torch.Tensor:
        img_pil = Image.open(image_path)
        img = np.array(img_pil, dtype='float')
        if len(img.shape) == 3:
            img = img[:, :, 0:3]
        else:
            img = img[:, :, None]
        mask = torch.from_numpy(img.transpose(2, 0, 1))
        if has_contour:
            none_class = torch.unique(mask)[-1]
            mask = np.where(mask == none_class, background_value, mask)
        return mask

    def load_mask_multi_channel(self,
                                image_path: str,
                                ignore_value: Optional[int] = None,
                                background_value: int = 0,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_pil = Image.open(image_path)
        img = np.array(img_pil)
        if len(img.shape) == 3:
            img = img[:, :, 0:3]
        else:
            img = img[:, :, None]

        if img.shape[2] == 3:
            # logging.warning(f"Image {image_path} has 3 channels RGB, but is expected to have 1 channel! Convert to grayscale!")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if len(img.shape) == 2:
                img = img[:, :, None]
            unique, vcount = np.unique(img, return_counts=True)
            if 255 in unique and vcount[unique == 255] > 0.5 * img.shape[0] * img.shape[1]:
                background_value = 255
        # Convert rgb mask to gray
        
        mask = torch.from_numpy(img.transpose(2, 0, 1))
        vals = torch.unique(mask)

        _valid_classes = torch.stack([x for x in vals if x != background_value and x != ignore_value])
        channel_mask = torch.zeros((len(_valid_classes),) + mask.squeeze().shape)
        for i, c in enumerate(_valid_classes):
            channel_mask[i, ...] = (mask == c)
            if ignore_value is not None:
                channel_mask = torch.where(mask == ignore_value, ignore_value, channel_mask)
        return channel_mask, _valid_classes

    @saveable()
    def plot(self, **kwargs) -> Figure:
        kwargs.update(dict(background_value=0))
        if 'scale' not in kwargs:
            kwargs['scale'] = 1
        if 'tight' not in kwargs:
            kwargs['tight'] = True
        mask = self.ground_truth_mask
        if mask is None:
            mask = torch.zeros(self.image.shape[1:])
        fig = plot_mask_labels(self.image, mask,
                               **kwargs
                               )
        return fig

    @saveable()
    def plot_selected(self, **kwargs) -> Figure:
        kwargs.update(dict(background_value=1))
        if 'scale' not in kwargs:
            kwargs['scale'] = 1
        if 'tight' not in kwargs:
            kwargs['tight'] = True
        mask = self.label
        if mask is None:
            mask = torch.zeros(self.image.shape[1:])
        fig = plot_mask_labels(self.image, mask,
                               **kwargs
                               )
        return fig

    @saveable()
    def plot_weak_labels(self,
                         all_object_ids: Optional[VEC_TYPE] = None,
                         **kwargs) -> Figure:
        if 'scale' not in kwargs:
            kwargs['scale'] = 1
        if 'tight' not in kwargs:
            kwargs['tight'] = True
        mask = self.trajectory_mask
        oid = self.trajectory_mask_object_ids
        if mask is None:
            mask = torch.zeros(self.image.shape[1:])
        if all_object_ids is not None:
            # All object ids are given, so we can map the weak labels to the given ids and preserve ordering
            super_mask = torch.zeros((len(all_object_ids),) + mask.shape[1:])
            for local_i, class_id in enumerate(oid):
                super_mask[all_object_ids == class_id, ...] = mask[local_i, ...]
            mask = super_mask
            oid = all_object_ids
        fig = plot_dense_image_mask(self.image, mask, oid,
                                    **kwargs
                                    )
        return fig

    @saveable()
    def plot_selected_weak_labels(self,
                                  **kwargs) -> Figure:
        if 'scale' not in kwargs:
            kwargs['scale'] = 1
        if 'tight' not in kwargs:
            kwargs['tight'] = True

        mask = self.weak_label
        c_mask, vals = value_mask_to_channel_masks(mask, background_value=self.WEAK_LABEL_NONECLASS)

        fig = plot_mask(self.image, c_mask, labels=[int(x) for x in vals],
                                    **kwargs,
                                    fill_contours=True
                                    )
        return fig

    @saveable()
    def plot_joint(self, **kwargs) -> Figure:
        fig = self.plot(**kwargs)
        ax = fig.axes[0]
        self.plot_weak_labels(ax=ax, plot_image=False)
        return fig

    @saveable()
    def plot_selected_joint(self, **kwargs) -> Figure:
        fig = self.plot_selected(**kwargs)
        ax = fig.axes[0]
        self.plot_selected_weak_labels(ax=ax, plot_image=False)
        return fig

    @saveable()
    def plot_trajectory_mask_with_ground_truth(self, size: float = 5, tight: bool = True) -> Figure:
        vals = self.trajectory_mask_object_ids
        
        gt = self.ground_truth_mask
        gt_ids = self._ground_truth_object_ids
        ax = None

        display_args = dict(
            tight=tight, size=size, 
        )

        if gt is not None:
            fig = plot_mask(self.image, gt, **display_args, darkening_background=0, labels=["GT: " + str(x.item()) for x in gt_ids])
            ax = fig.axes[0]
        fig = plot_mask(self.image if gt is None else None, self.trajectory_mask, ax=ax, **display_args, filled_contours=True, labels=["TJ: " + str(x.item()) for x in vals])
        return fig