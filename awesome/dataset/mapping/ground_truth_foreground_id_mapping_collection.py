from dataclasses import dataclass
from typing import Dict

from awesome.dataset.mapping.ground_truth_foreground_id_mapping import GroundTruthForegroundIdMapping
from awesome.serialization.json_convertible import JsonConvertible

from dataclasses import dataclass, field

@dataclass
class GroundTruthForegroundIdMappingCollection(JsonConvertible):
    """Stores for all sequences within fbms the mapping."""

    mappings : Dict[int, Dict[str, "GroundTruthForegroundIdMapping"]] = field(default_factory=dict)
    """Defines mappings for individual sequences, also for multiple objects."""

