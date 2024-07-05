


from typing import Any, Dict, Optional
import torch

class MemoryNode():

    name: str

    parent: Optional['MemoryNode']

    children: Dict[str, Any]

    reference_object: Any

    reference_object_python_size: int

    reference_object_device_size: int

    reference_object_device: torch.device

    total_python_size: int

    total_device_size: Dict[torch.device, int]

    # def __init__(self, 
    #              parent: Optional['MemoryNode']
    #              children: 
    #              ) -> None:
    #     pass