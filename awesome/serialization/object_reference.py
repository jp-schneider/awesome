from dataclasses import dataclass
from typing import Any, Dict
from uuid import UUID
from awesome.util.reflection import class_name


@dataclass
class ObjectReference:
    """A object reference marking a already serialized object."""

    uuid: str = None
    """Object identifier of the original object."""

    object_type: str = None
    """Original object type which was referenced."""

    def to_dict(self) -> Dict[str, Any]:
        return dict(__class__=class_name(ObjectReference),
                    uuid=self.uuid,
                    object_type=self.object_type)

    def __hash__(self) -> str:
        return int(hash(self.uuid) + hash(self.object_type))