from functools import lru_cache
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from matplotlib.figure import Figure
import pandas as pd
import torch
import os
from tqdm.autonotebook import tqdm
from awesome.dataset.fbms_sequence_sample import FBMSSequenceSample
from awesome.dataset.mapping.ground_truth_foreground_id_mapping_collection import GroundTruthForegroundIdMappingCollection
from awesome.dataset.torch_datasource import TorchDataSource
import copy
from PIL import Image
import numpy as np
from awesome.dataset.trajectory_util.ground_truth_description_file import GroundTruthDescriptionFile
from awesome.dataset.trajectory_util.trajectory import Trajectory

from awesome.run.functions import channel_masks_to_value_mask, save_mask, saveable, value_mask_to_channel_masks
from awesome.serialization.json_convertible import JsonConvertible
from awesome.util.format import parse_enum

import re
from typing import List
import numpy as np
from awesome.util.path_tools import format_os_independent
from awesome.util.temporary_property import TemporaryProperty
import shutil
from awesome.dataset.trajectory_util.frame_description import FrameDescription
from awesome.dataset.mapping.ground_truth_foreground_id_mapping import GroundTruthForegroundIdMapping
from awesome.dataset.trajectory_util.mask_generation_object import MaskGenerationObject
from awesome.dataset.label_mode import LabelMode


def plot_nan_masks(masks: torch.Tensor):
    path = "temp"
    import matplotlib.pyplot as plt
    import os
    non_containing_masks = [(i, masks[i]) for i in range(
        masks.shape[0]) if torch.isnan(masks[i]).any()]
    for index, mask in non_containing_masks:
        fig = plt.figure()
        plt.imshow(mask.detach().cpu().numpy())
        plt.title(f"Mask {index}")
        plt.colorbar()
        fig.savefig(os.path.join(path, f"mask_{index}.png"))
        plt.close(fig)


def create_index_df():
    return pd.DataFrame(columns=[x for x in FBMSSequenceSample.__dataclass_fields__.keys()])


class FBMSSequenceDataset(TorchDataSource):

    __samples__: pd.DataFrame
    """The index of images to load."""

    ground_truth_dir: str = "GroundTruth"

    image_dir: str = "."

    weak_labels_dir: str

    confidence_dir: Optional[str]

    trajectories_dir: str
    """Directory where the trajectory file is located."""

    dataset_path: str

    annotations_file: str = "annotations.json"

    converted_trajectories_file: str = "converted_trajectories.json"

    _annotations: GroundTruthDescriptionFile

    _trajectories: pd.DataFrame

    def __init__(self,
                 dataset_path: str = None,
                 all_frames: bool = False,
                 use_memory_cache: bool = False,
                 compressed_dataset: bool = True,
                 trajectory_dir: str = "tracks/multicut",
                 trajectory_file: Optional[str] = None,
                 weak_labels_dir: str = "weak_labels",
                 processed_weak_labels_dir: str = "processed_weak_labels",
                 do_weak_label_preprocessing: bool = False,
                 do_uncertainty_label_flip: bool = False,
                 confidence_dir: Optional[str] = None,
                 test_weak_label_integrity: bool = True,
                 remove_cache: bool = False,
                 dtype: torch.dtype = torch.float32,
                 decoding: bool = False,
                 split_ratio: float = 1,
                 _no_indexing: bool = False,
                 segmentation_object_mapping_file: str = "data/fbms_segmentation_object_mapping.json",
                 segmentation_object_id: Union[int, List[int]] = 0,
                 label_mode: LabelMode = LabelMode.SINGLE_OBJECT,
                 **kwargs
                 ):
        super().__init__(returns_index=False, split_ratio=split_ratio, **kwargs)
        self.use_memory_cache = use_memory_cache
        self.compressed_dataset = compressed_dataset
        self.trajectories_dir = trajectory_dir
        self.trajectories_file = trajectory_file
        self.all_frames = all_frames
        self.weak_labels_dir = weak_labels_dir
        self.processed_weak_labels_dir = processed_weak_labels_dir
        self.do_weak_label_preprocessing = do_weak_label_preprocessing
        self.do_uncertainty_label_flip = do_uncertainty_label_flip
        self.confidence_dir = confidence_dir
        self.test_weak_label_integrity = test_weak_label_integrity
        self._annotations = None
        self._trajectories = None
        self.remove_cache = remove_cache
        self.__index__ = None
        self.dtype = dtype
        self.segmentation_object_mapping_file = segmentation_object_mapping_file
        self.segmentation_object_id = segmentation_object_id
        self.label_mode = parse_enum(LabelMode, label_mode)
        self.__weak_label_ground_truth_mappings__ = None

        if decoding:
            return
        self.dataset_path = format_os_independent(
            os.path.normpath(dataset_path))
        self.__index__ = None
        if not _no_indexing:
            self.__index__ = self.index()

    @property
    def dataset_name(self) -> str:
        """Name of the dataset => Folder of the dataset path."""
        return os.path.basename(self.dataset_path)

    def get_number_of_objects(self) -> int:
        """Gets the number of foreground objects in the the current sequence, based on what the weak labels suggests.

        Returns
        -------
        int
            Number of objects.
        """
        self._assure_indexed()
        df = self.__index__
        sample = FBMSSequenceSample.from_series(df.iloc[0])
        if isinstance(sample.foreground_weak_label_object_id, int):
            return 1
        else:
            return len(sample.foreground_weak_label_object_id)

    def get_segmentation_object_mapping(self, generate_masks: bool, mgo: "MaskGenerationObject") -> "GroundTruthForegroundIdMapping":
        collection = None
        mapping = None
        _save = False
        if not os.path.exists(self.segmentation_object_mapping_file):
            os.makedirs(os.path.dirname(
                self.segmentation_object_mapping_file), exist_ok=True)
            collection = GroundTruthForegroundIdMappingCollection()
            _save = True
        else:
            collection = GroundTruthForegroundIdMappingCollection.load_from_file(
                self.segmentation_object_mapping_file)
        if self.label_mode == LabelMode.SINGLE_OBJECT:
            if not isinstance(self.segmentation_object_id, int):
                raise ValueError(
                    "Label mode is single object, but segmentation object id is not an integer!")

            oid = str(self.segmentation_object_id)
            if oid not in collection.mappings:
                collection.mappings[oid] = {}
                _save = True
            if self.dataset_name not in collection.mappings[oid]:
                mapping = self.get_default_ground_truth_mapping(
                    generate_masks=generate_masks, mgo=mgo)
                collection.mappings[oid][self.dataset_name] = mapping
                _save = True
            else:
                mapping = collection.mappings[oid][self.dataset_name]
        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            mapping = []
            if not isinstance(self.segmentation_object_id, list):
                raise ValueError(
                    "Label mode is multi object, but segmentation object id is not a list!")

            if len(self.segmentation_object_id) == 0:
                # Empty list means all objects
                # All objects
                default_mappings = self.get_ground_truth_id_mapping_across_all_frames(
                    generate_masks=generate_masks, mgo=mgo)
                for oid, _mapping in enumerate(default_mappings):
                    if str(oid) not in collection.mappings:
                        collection.mappings[str(oid)] = {}
                        _save = True
                    if self.dataset_name not in collection.mappings[str(oid)]:
                        collection.mappings[str(
                            oid)][self.dataset_name] = _mapping
                        _save = True
                        mapping.append(_mapping)
                    else:
                        mapping.append(
                            collection.mappings[str(oid)][self.dataset_name])
            else:
                # Specific objects should be selected. Select them based on there trajectory id.
                mapping = []
                default_mappings = None
                for oid in self.segmentation_object_id:
                    if str(oid) not in collection.mappings:
                        collection.mappings[str(oid)] = {}
                        _save = True
                    if self.dataset_name not in collection.mappings[str(oid)]:
                        if default_mappings is None:
                            default_mappings = self.get_ground_truth_id_mapping_across_all_frames(
                                generate_masks=generate_masks, mgo=mgo)
                        _mapping = default_mappings[oid]
                        collection.mappings[str(
                            oid)][self.dataset_name] = mapping
                        _save = True
                    else:
                        _mapping = collection.mappings[str(
                            oid)][self.dataset_name]
                    mapping.append(_mapping)

        else:
            raise ValueError(f"Label mode: {self.label_mode} not supported!")

        if _save:
            collection.save_to_file(
                self.segmentation_object_mapping_file, override=True)

        return mapping

    def get_default_ground_truth_mapping(self, generate_masks: bool, mgo: "MaskGenerationObject") -> Union["GroundTruthForegroundIdMapping", List["GroundTruthForegroundIdMapping"]]:
        """Gets the default ground truth mapping for the dataset.

        This will be the the mapping indicated by the first annotated frame.

        This is the mapping from the segmentation object as suggested by the weak label / trajectory mask and the ground truth object masks.

        In single object mode, the mapping is a single object, in multi object mode, the mapping is a list of objects.

        Parameters
        ----------
        generate_masks : bool
            If true, the masks are generated.
            If false, the masks are not generated based on the trajectory file, so its expected that the masks are already generated.

        mgo : MaskGenerationObject
            Mask generation object.

        Returns
        -------
        GroundTruthForegroundIdMapping
            Mapping object.
        """

        frame_descriptions = self.get_frame_descriptions()
        # Pre determine foreground object id based on first annotated image
        # For matching with trajectory masks as size based could change it.
        idx, first_gt_frame = next(((i, x) for i, x in enumerate(
            frame_descriptions) if x.ground_truth_file_name is not None), None)
        return self.get_ground_truth_id_mapping_for_frame(idx, first_gt_frame, generate_masks=generate_masks, mgo=mgo)

    def get_ground_truth_id_mapping_across_all_frames(self, generate_masks: bool, mgo: "MaskGenerationObject") -> Union["GroundTruthForegroundIdMapping", List["GroundTruthForegroundIdMapping"]]:
        frame_descriptions = self.get_frame_descriptions()
        object_id_mappings = dict()
        final_mappings = list()
        # Get the mappings for all frames
        for (i, x) in enumerate(frame_descriptions):
            if x.ground_truth_file_name is None:
                continue
            mapping = self.get_ground_truth_id_mapping_for_frame(
                i, x, generate_masks=generate_masks, mgo=mgo)
            if self.label_mode == LabelMode.SINGLE_OBJECT:
                if 0 not in object_id_mappings:
                    object_id_mappings[0] = []
                object_id_mappings[0].append(mapping)
            elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
                for oidx, m in enumerate(mapping):
                    if oidx not in object_id_mappings:
                        object_id_mappings[oidx] = []
                    object_id_mappings[oidx].append(m)
        # Vote for the most common mapping per object
        for oidx, mappings in object_id_mappings.items():
            background_id = 0
            maps = np.array([(x.trajectory_foreground_id, x.ground_truth_foreground_id)
                            for x in mappings if x.ground_truth_foreground_id != background_id and x.ground_truth_foreground_id != None], dtype=np.int32)
            unique, counts = np.unique(maps, axis=0, return_counts=True)
            most_common = unique[np.argmax(counts)]
            most_common_map = next((x for x in mappings if x.trajectory_foreground_id ==
                                   most_common[0] and x.ground_truth_foreground_id == most_common[1]))
            final_mappings.append(most_common_map)
        if len(final_mappings) == 1 and self.label_mode == LabelMode.SINGLE_OBJECT:
            return final_mappings[0]
        return final_mappings

    def get_ground_truth_id_mapping_for_frame(self, index: int, frame_description: FrameDescription, generate_masks: bool, mgo: "MaskGenerationObject") -> Union["GroundTruthForegroundIdMapping", List["GroundTruthForegroundIdMapping"]]:
        """Gets the ground truth id mapping for the given frame.

        This is the mapping from the segmentation object as suggested by the weak label / trajectory mask and the ground truth object masks.

        In single object mode, the mapping is a single object, in multi object mode, the mapping is a list of objects.

        Parameters
        ----------
        index : int
            The id of the frame to get the mapping for.

        frame_description : FrameDescription
            The frame description.

        generate_masks : bool
            If true, the masks are generated.
            If false, the masks are not generated based on the trajectory file, so its expected that the masks are already generated.

        mgo : MaskGenerationObject
            Mask generation object.

        Returns
        -------
        GroundTruthForegroundIdMapping
            Mapping object.
        """
        sample = self._load_sample(
            index, frame_description, generate_masks=generate_masks, mgo=mgo)

        # Access weak label to assure it is loaded
        _ = sample.weak_label

        if self.label_mode == LabelMode.SINGLE_OBJECT:
            foreground_weak_label_object_id = sample.foreground_weak_label_object_id
            foreground_gt_object_id = sample.foreground_gt_object_id
            return GroundTruthForegroundIdMapping(
                dataset_name=self.dataset_name,
                trajectory_foreground_id=foreground_weak_label_object_id,
                ground_truth_foreground_id=foreground_gt_object_id)

        elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
            foreground_weak_label_object_ids = sample.foreground_weak_label_object_id
            foreground_gt_object_ids = sample.foreground_gt_object_id

            ret = []
            for wlid, gtid in zip(foreground_weak_label_object_ids, foreground_gt_object_ids):
                ret.append(GroundTruthForegroundIdMapping(
                    dataset_name=self.dataset_name,
                    trajectory_foreground_id=wlid,
                    ground_truth_foreground_id=gtid))

            # Sort them based on there trajectory id
            ret = sorted(ret, key=lambda x: x.trajectory_foreground_id)
            return ret
        else:
            raise ValueError(f"Label mode: {self.label_mode} not supported!")

    @staticmethod
    def trajectories_frame_to_mask(traj_df: pd.DataFrame, frame_id: int, image_shape: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a frame of the trajectory dataframe to a mask for the given frame id.
        Works in combination with the ``get_framed_based_weak_labels`` function.

        Parameters
        ----------
        traj_df : pd.DataFrame
            The frame based trajectory dataframe.
        frame_id : int
            The id of the frame to create a mask for.
        image_shape : tuple
            The shape of the image. (C, H, W) or (H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The mask of the frame. In the form (N, H, W) where N is the number of unique object ids in the frame.
            2. The unique object ids in the frame. In the form (N,).
        """
        c, h, w = None, None, None
        if len(image_shape) == 3:
            c, h, w = image_shape
        elif len(image_shape) == 2:
            h, w = image_shape
            c = 1
        else:
            raise ValueError("Image shape must be 2 or 3 dimensional!")

        df = traj_df[traj_df["frame_ids"] ==
                     frame_id][['coordinates', 'object_id']]
        unique_object_ids = torch.tensor(df['object_id'].unique())

        mask = torch.zeros((len(unique_object_ids),) + (h, w))
        for i, object_id in enumerate(unique_object_ids):
            object_coordinates = np.stack(
                df[df['object_id'] == object_id.item()]['coordinates'].values)
            x_coord, y_coord = np.round(object_coordinates[:, 0]), np.round(
                object_coordinates[:, 1])
            mask[i, y_coord, x_coord] = 1

        return mask, unique_object_ids

    def get_framed_based_weak_labels(self) -> pd.DataFrame:
        """Gets a dataframe with all weak labels defined in the trajectory file.
        The columns are: frame_id, object_id, coordinates, line_start

        Returns
        -------
        pd.DataFrame
            Dataframe
        """
        trajectories = self.trajectories
        return trajectories.explode(['coordinates', 'frame_ids'])

    @property
    def image_path(self) -> str:
        return os.path.join(self.dataset_path, self.image_dir)

    @property
    def ground_truth_path(self) -> str:
        return os.path.join(self.dataset_path, self.ground_truth_dir)

    @property
    def weak_labels_path(self) -> str:
        return os.path.join(self.dataset_path, self.weak_labels_dir)

    @property
    def processed_weak_labels_path(self) -> str:
        return os.path.join(self.dataset_path, self.processed_weak_labels_dir)

    @property
    def confidence_path(self) -> str:
        return os.path.join(self.dataset_path, self.confidence_dir)

    @property
    def sequence_name(self) -> str:
        return str(os.path.split(self.dataset_path)[1])

    @property
    def definition_file_path(self) -> str:
        def_file_name = self.sequence_name + "Def.dat"
        return os.path.join(self.dataset_path, self.ground_truth_dir, def_file_name)

    @property
    def annotations_path(self) -> str:
        return os.path.join(self.dataset_path, self.annotations_file)

    @property
    def trajectories_file_path(self) -> str:
        path = format_os_independent(os.path.normpath(
            os.path.join(self.dataset_path, self.trajectories_dir)))
        if self.trajectories_file is not None:
            path = os.path.join(path, self.trajectories_file)
        else:
            file = os.listdir(path)[0]
            path = os.path.join(path, file)
        return path

    @property
    def converted_trajectories_file_path(self) -> str:
        return os.path.join(self.dataset_path, self.converted_trajectories_file)

    @property
    @lru_cache(maxsize=32)
    def trajectories(self) -> pd.DataFrame:
        if self._trajectories is None:
            traj = pd.DataFrame(self._get_trajectories())
            if not self.use_memory_cache:
                return traj
            self._trajectories = traj
        return self._trajectories

    @property
    def annotations(self) -> GroundTruthDescriptionFile:
        if self._annotations is None:
            annotations = self._get_annotations()
            if not self.use_memory_cache:
                return annotations
            self._annotations = annotations
        return self._annotations

    @property
    def unique_weak_label_object_ids(self) -> torch.Tensor:
        """Returns all unique object ids in the weak label frames.

        Returns
        -------
        torch.Tensor
            Tensor of all unique object ids (N,).
        """
        ids = [self[i].trajectory_mask_object_ids for i in range(len(self))]
        return torch.stack([x for y in ids for x in y]).unique()

    def parse_annotations(self) -> GroundTruthDescriptionFile:
        file = None
        with open(self.definition_file_path, "r") as f:
            file = f.read()
        return GroundTruthDescriptionFile.from_str(file, self.compressed_dataset)

    def _get_annotations(self) -> GroundTruthDescriptionFile:
        annotations = None
        if not os.path.exists(self.annotations_path):
            annotations = self.parse_annotations()
            annotations.save_to_file(self.annotations_path, no_uuid=True)
        else:
            annotations = GroundTruthDescriptionFile.load_from_file(
                self.annotations_path)
        return annotations

    def __ignore_on_iter__(self) -> Set[str]:
        # ret = super().__ignore_on_iter__()
        ret = set()
        ret.add("__index__")
        return ret

    def _assure_indexed(self, test_weak_label_integrity: bool = False) -> None:
        if self.__index__ is None:
            with TemporaryProperty(self, test_weak_label_integrity=test_weak_label_integrity):
                self.__index__ = self.index()

    def get_ground_truth_indices(self) -> List[int]:
        """Get the indices of all samples hich habe ground truth."""
        # If index is not set, then index but dont test the weak label integrity
        self._assure_indexed()
        df = self.__index__
        cand = df[df['has_label'] == True]['index'].values.tolist()
        return [x for x in cand if FBMSSequenceSample.from_series(df.iloc[x]).is_ground_truth_evaluatable]

    def _generate_masks(self, frame: FrameDescription, mgo: MaskGenerationObject) -> None:
        mask_file_name = os.path.splitext(frame.image_file_name)[0] + ".png"
        mask_path = os.path.join(self.weak_labels_path, mask_file_name)

        frame_trajectories = mgo.frame_trajectories if mgo is not None else None
        img_shape = mgo.img_shape if mgo is not None else None

        if frame_trajectories is None:
            frame_trajectories = self.get_framed_based_weak_labels()
            if mgo is not None:
                mgo.frame_trajectories = frame_trajectories

        if img_shape is None:
            img_path = os.path.join(self.image_path, frame.image_file_name)
            img = Image.open(img_path)
            img_shape = np.array(img).shape
            if mgo is not None:
                mgo.img_shape = img_shape

        channel_mask, unique_object_ids = FBMSSequenceDataset.trajectories_frame_to_mask(
            frame_trajectories, int(frame.frame_number), img_shape[:2])
        # Map 0 Class to 255
        if 0 in unique_object_ids:
            unique_object_ids[unique_object_ids == 0] = 255
        value_mask = channel_masks_to_value_mask(
            channel_mask, unique_object_ids, "warning+exclude")
        save_mask(value_mask, mask_path)

    def _load_sample(self,
                     index: int,
                     frame: FrameDescription,
                     generate_masks: bool = False,
                     mgo: Optional[MaskGenerationObject] = None,
                     foreground_weak_label_object_id: Optional[int] = None,
                     foreground_gt_object_id: Optional[int] = None,
                     ) -> FBMSSequenceSample:
        mask_file_name = os.path.splitext(frame.image_file_name)[0] + ".png"
        mask_path = os.path.join(self.weak_labels_path, mask_file_name)

        if generate_masks or not os.path.exists(mask_path):
            self._generate_masks(frame=frame, mgo=mgo)

        gt_path = os.path.normpath(os.path.join(self.ground_truth_path, frame.ground_truth_file_name)
                                   ) if frame.ground_truth_file_name is not None else None
        preprocessed_label_path = None
        confidence_path = None
        if self.processed_weak_labels_dir is not None:
            preprocessed_label_path = os.path.join(
                self.processed_weak_labels_path, mask_file_name)
        if self.confidence_dir is not None:
            confidence_file_name = os.path.splitext(frame.image_file_name)[
                0] + "_confidence" + ".h5"
            confidence_path = os.path.join(
                self.confidence_path, confidence_file_name)

        sample = FBMSSequenceSample(
            sequence_name=self.sequence_name,
            index=frame.frame_number,
            image_path=os.path.normpath(os.path.join(
                self.image_path, frame.image_file_name)),
            ground_truth_path=gt_path,
            trajectory_mask_path=mask_path,
            processed_weak_label_path=preprocessed_label_path,
            do_weak_label_preprocessing=self.do_weak_label_preprocessing,
            do_uncertainty_label_flip=self.do_uncertainty_label_flip,
            confidence_file_path=confidence_path,
            has_label=True if gt_path is not None else False,
            foreground_weak_label_object_id=foreground_weak_label_object_id,
            foreground_gt_object_id=foreground_gt_object_id,
            label_mode=self.label_mode,
        )
        if self.test_weak_label_integrity:
            try:
                _ = sample.weak_label
            except Exception as err:
                logging.warning(
                    f"Error in getting weak label in frame {frame.frame_number}, path {sample.trajectory_mask_path}! Excluding frame! \n {err}")
                return None
            if not sample.has_valid_foreground_and_background:
                logging.warning(
                    f"Frame {frame.frame_number} has no valid foreground and background! Excluding frame!")
                return None
        return sample

    def get_frame_descriptions(self) -> List[FrameDescription]:
        """Gets the frame descriptions of the dataset.

        Returns
        -------
        List[FrameDescription]
            All frame descriptions.
        """
        frame_descriptions = self.annotations.frame_descriptions
        # Load all frames
        if self.all_frames:
            smallest_index = min([x.frame_number for x in frame_descriptions])
            smallest_name = next(
                (x.frame_name for x in frame_descriptions if x.frame_number == smallest_index), None)
            name_index_mapping = {k: v for k, v in zip(range(smallest_name, smallest_name + self.annotations.total_number_of_frames),
                                                       range(smallest_index, smallest_index + self.annotations.total_number_of_frames))}
            existing = set([x.frame_name for x in frame_descriptions])
            out_frames = self.read_directory(self.image_path, FrameDescription.IMAGE_NAME_PATTERN, keys=[
                                             "dataset_name", "frame_name"], primary_key="frame_name")
            add = [FrameDescription(frame_number=name_index_mapping[int(k)], frame_name=int(
                k), image_file_name=os.path.basename(v["path"])) for k, v in out_frames.items() if int(k) not in existing]
            frame_descriptions.extend(add)
        # Sort
        frame_descriptions = sorted(
            frame_descriptions, key=lambda x: x.frame_number)
        return frame_descriptions

    def index(self) -> pd.DataFrame:
        frames = []

        generate_masks = False
        mgo = MaskGenerationObject()

        if not (os.path.exists(self.weak_labels_path) and len(os.listdir(self.weak_labels_path)) != 0):
            os.makedirs(self.weak_labels_path, exist_ok=True)
            generate_masks = True
            frame_trajectories = self.get_framed_based_weak_labels()
            mgo.frame_trajectories = frame_trajectories

        frame_descriptions = self.get_frame_descriptions()

        if self.remove_cache and os.path.exists(self.processed_weak_labels_path):
            shutil.rmtree(self.processed_weak_labels_path, ignore_errors=True)
            os.makedirs(self.processed_weak_labels_path, exist_ok=True)

        mapping = None
        try:
            mapping = self.get_segmentation_object_mapping(
                generate_masks=generate_masks, mgo=mgo)
            self.__weak_label_ground_truth_mappings__ = mapping
        except Exception as err:
            logging.warning(
                f"Error in getting segmentation object mapping! {str(err)}")

        for idx, frame in enumerate(tqdm(frame_descriptions, desc="Loading frames...", delay=2)):
            foreground_weak_label_object_id = None
            foreground_gt_object_id = None
            if self.label_mode == LabelMode.SINGLE_OBJECT:
                foreground_weak_label_object_id = mapping.trajectory_foreground_id if mapping is not None else None
                foreground_gt_object_id = mapping.ground_truth_foreground_id if mapping is not None else None

            elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
                foreground_weak_label_object_id = [
                    x.trajectory_foreground_id for x in mapping] if mapping is not None else None
                foreground_gt_object_id = [
                    x.ground_truth_foreground_id for x in mapping] if mapping is not None else None
            else:
                raise ValueError(
                    f"Label mode: {self.label_mode} not supported!")

            sample = self._load_sample(idx, frame,
                                       generate_masks=generate_masks, mgo=mgo,
                                       foreground_weak_label_object_id=foreground_weak_label_object_id,
                                       foreground_gt_object_id=foreground_gt_object_id
                                       )
            if sample is not None:
                frames.append(sample)
        df = pd.DataFrame([x.to_series() for x in frames])
        return df

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
                entry = dict(path=os.path.join(path, file), **data)
                res[entry[primary_key]] = entry
        return res

    def __getitem__(self, index: int):
        if self.__index__ is None:
            self.__index__ = self.index()
        row = self.__index__.iloc[index]
        return FBMSSequenceSample.from_series(row)

    def __len__(self):
        if self.__index__ is None:
            self.__index__ = self.index()
        return len(self.__index__)

    def _get_trajectories(self) -> List[Trajectory]:
        if os.path.exists(self.converted_trajectories_file_path):
            return JsonConvertible.load_from_file(self.converted_trajectories_file_path)
        else:
            trajectories = FBMSSequenceDataset.parse_trajectories_file(
                self.trajectories_file_path)
            JsonConvertible.convert_to_json_file(
                trajectories, self.converted_trajectories_file_path, no_uuid=True)
            return trajectories

    @staticmethod
    def parse_trajectories_file(path: str) -> List[Trajectory]:

        with open(path, "r") as f:
            lines = f.readlines()

        trajectories = []

        num_trajectories = None
        active_trajectory = False

        current_trajectory_object_id = -1
        current_trajectory_coordinates = None
        current_trajectory_frame_id = None
        current_trajectory_len = -1
        current_trajectory_start_line = -1

        for i, line in enumerate(lines):
            if i == 0:
                continue
            if i == 1:
                num_trajectories = int(line)
                continue
            if active_trajectory:
                values = line.replace("\n", "").split(" ")
                current_trajectory_coordinates.append(
                    [float(x) for x in values[:-1]])
                current_trajectory_frame_id.append(int(values[-1]))
                if len(current_trajectory_coordinates) == current_trajectory_len:
                    active_trajectory = False
                    coordinates = np.array(current_trajectory_coordinates)
                    frame_ids = np.array(current_trajectory_frame_id)
                    trajectories.append(Trajectory(object_id=current_trajectory_object_id,
                                                   coordinates=coordinates,
                                                   frame_ids=frame_ids, line_start=current_trajectory_start_line))
                    current_trajectory_coordinates = None
                    current_trajectory_frame_id = None
                    current_trajectory_len = -1
                    current_trajectory_start_line = -1
                    current_trajectory_object_id = -1
            else:
                # Start of new trajectory
                values = line.replace("\n", "").split(" ")
                current_trajectory_object_id = int(values[0])
                current_trajectory_len = int(values[1])
                current_trajectory_start_line = i + 1
                current_trajectory_coordinates = []
                current_trajectory_frame_id = []
                active_trajectory = True

        assert len(
            trajectories) == num_trajectories, f"Number of trajectories does not match: Should: {num_trajectories} but got {len(trajectories)}"
        return trajectories

    @saveable()
    def plot_ground_truth_mask_images(self, size: float = 5, tight: bool = False) -> Figure:
        """Plots all ground truth frames with the mask and trajectories.

        Parameters
        ----------
        size : float, optional
            Size of the individual plots, by default 5
        tight : bool, optional
            If plots should be placed tight, by default False

        Returns
        -------
        Figure
            Figure
        """
        from awesome.run.functions import get_mpl_figure, plot_mask

        def add_label_info(_id, sample, mode: Literal["weak", "gt"]) -> str:
            if isinstance(_id, torch.Tensor):
                _id = _id.item()
            if self.label_mode == LabelMode.SINGLE_OBJECT:
                if mode == "weak":
                    if _id == sample.foreground_weak_label_object_id:
                        return str(_id) + " FG"
                    elif _id == sample.background_weak_label_object_id:
                        return str(_id) + " BG"
                    else:
                        return str(_id)
                elif mode == "gt":

                    gt_fg = sample.weak_label_id_ground_truth_object_id_mapping.get(
                        sample.foreground_weak_label_object_id, None)
                    gt_bg = sample.weak_label_id_ground_truth_object_id_mapping.get(
                        sample.background_weak_label_object_id, None)

                    if _id == gt_fg:
                        return str(_id) + " FG"
                    elif _id == gt_bg:
                        return str(_id) + " BG"
                    else:
                        return str(_id)

            elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
                if mode == "weak":
                    if _id in sample.foreground_weak_label_object_id:
                        return str(_id) + " FG"
                    elif _id == sample.background_weak_label_object_id:
                        return str(_id) + " BG"
                    else:
                        return str(_id)
                elif mode == "gt":
                    wk_labels = sample.ground_truth_object_id_weak_label_mapping.get(
                        _id, None)
                    used_as_fg = any(
                        [x in sample.foreground_weak_label_object_id for x in wk_labels])
                    gt_bg = sample.weak_label_id_ground_truth_object_id_mapping.get(
                        sample.background_weak_label_object_id)

                    if used_as_fg:
                        return str(_id) + f" FG"
                    elif _id == gt_bg:
                        return str(_id) + " BG"
                    else:
                        return str(_id)

        indices = self.get_ground_truth_indices()

        rows = len(indices)
        fig, axs = get_mpl_figure(
            rows=rows, cols=3, size=size, tight=tight, ratio_or_img=self[0].image, ax_mode="2d")

        for i, index in enumerate(indices):
            sample = self[index]

            try:
                _ = sample.weak_label
            except Exception:
                pass

            row_axs = axs[i]
            fig = plot_mask(sample.image, sample.trajectory_mask, ax=row_axs[0],
                            filled_contours=True,
                            labels=[add_label_info(x, sample, "weak") for x in sample.trajectory_mask_object_ids])
            row_axs[0].set_title("Weak Label: " + str(index))

            try:
                fig = plot_mask(sample.image, sample.ground_truth_mask, ax=row_axs[1], labels=[
                                add_label_info(x, sample, "gt") for x in sample.ground_truth_object_ids])
                row_axs[1].set_title("GT: " + str(index))
            except Exception:
                pass

            try:
                if self.label_mode == LabelMode.SINGLE_OBJECT:
                    fig = plot_mask(sample.image, 1 -
                                    sample.label, ax=row_axs[2])
                elif self.label_mode == LabelMode.MULTIPLE_OBJECTS:
                    fig = plot_mask(sample.image, value_mask_to_channel_masks(sample.label)[
                                    0], ax=row_axs[2], labels=[str(x) for x in sample.foreground_gt_object_id])
                row_axs[2].set_title("GT Selected: " + str(index))
            except Exception:
                pass
        return fig
