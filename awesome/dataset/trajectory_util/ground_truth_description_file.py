from dataclasses import dataclass
import re
from typing import List
import numpy as np

from awesome.serialization.json_convertible import JsonConvertible

from dataclasses import dataclass, field
import re
from typing import List
import numpy as np
from awesome.dataset.trajectory_util.frame_description import FrameDescription


@dataclass
class GroundTruthDescriptionFile(JsonConvertible):

    PARSING_PATTERN = (r"^(?P<preamble>[\w+ ;!]+)(\r)?\n(\r)?\n(Total number of regions:(\r)?\n"
                       + r"(?P<total_number_of_regions>[\d]+))(\r)?\n(?P<scale_of_regions>((Scale of region [\d]+:)"
                       + r"(\r)?\n([\d]+)(\r)?\n)+)(\r)?\n(Confusion penality matrix:(\r)?\n"
                       + r"(?P<confusion_penality_matrix>((((([\d]+(\.\d+)?( )*)+)))(\r)?\n)*))((\r)?\n)?"
                       + r"(Total number of frames in this shot:)(\r)?\n(?P<total_number_of_frames>[\d]+)(\r)?\n"
                       + r"(Total number of labeled frames for this shot:)(\r)?\n(?P<total_number_of_labeled_frames>[\d]+)(\r)?\n"
                       + r"(?P<frame_descriptions>[\w\s\:\.]+)$")

    SCALE_OF_REGION_PATTERN = r"(Scale of region (?P<region>[\d]+):)(\r)?\n(?P<scale>[\d]+)"

    total_number_of_regions: int = field(default=None)

    scale_of_regions: np.ndarray = field(default=None)

    confusion_penality_matrix: np.ndarray = field(default=None)

    total_number_of_frames: int = field(default=None)

    total_number_of_labeled_frames: int = field(default=None)

    frame_descriptions: List['FrameDescription'] = field(default=None)

    @staticmethod
    def from_str(content: str, compressed_dataset: bool = False):
        groups = GroundTruthDescriptionFile.__dataclass_fields__.keys()
        m = re.match(GroundTruthDescriptionFile.PARSING_PATTERN, content)
        if m is None:
            raise Exception("Could not parse content")
        items = dict()
        for g in groups:
            content = m.group(g)
            if content is None:
                raise Exception(f"Could not parse group {g}")
            items[g] = content
        int_fields = ["total_number_of_regions", "total_number_of_frames", "total_number_of_labeled_frames"]
        for f in int_fields:
            items[f] = int(items[f])

        mat_str = items["confusion_penality_matrix"]
        mat_rows = mat_str.split("\n")
        mat_items = [[float(y) for y in x.strip().split(" ") if y != ""] for x in mat_rows if x != ""]
        mat = np.array(mat_items)
        items["confusion_penality_matrix"] = mat

        scale_str = items["scale_of_regions"]
        regions_match = re.finditer(GroundTruthDescriptionFile.SCALE_OF_REGION_PATTERN, scale_str)
        scale_of_regions = np.zeros(items["total_number_of_regions"], dtype=np.int32)
        regions = []
        for r_m in regions_match:
            scale = int(r_m.group("scale"))
            region = int(r_m.group("region"))
            scale_of_regions[region] = scale
        items["scale_of_regions"] = scale_of_regions

        frame_descriptions = []
        frame_matches = re.finditer(FrameDescription.TRAJECTORY_FILE_PARSING_PATTERN, items["frame_descriptions"])
        for m in frame_matches:
            groups = ["frame_number", "ground_truth_file_name", "image_file_name"]
            frame_items = dict()
            for g in groups:
                content = m.group(g)
                if content is None:
                    raise Exception(f"Could not parse group {g}")
                frame_items[g] = content
            int_fields = ["frame_number"]
            for f in int_fields:
                frame_items[f] = int(frame_items[f])
            im_match = re.match(FrameDescription.IMAGE_NAME_PATTERN, frame_items["image_file_name"])
            if im_match is not None:
                frame_items["frame_name"] = int(im_match.group("frame_name"))
            desc = FrameDescription(**frame_items)
            if compressed_dataset:
                desc.image_file_name = desc.image_file_name.replace(".ppm", ".jpg")
            frame_descriptions.append(desc)
        items["frame_descriptions"] = frame_descriptions

        return GroundTruthDescriptionFile(**items)
