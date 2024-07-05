

from typing import Optional
from torch.nn.modules import Module
from awesome.model.abstract_combined_segmentation_module import AbstractCombinedSegmentationModule, EvaluationMode, GradientMode, InputMode, PriorMode, PriorOutputProcessingMode, SegmentationOutputProcessingMode


class CombinedSegmentationModule(AbstractCombinedSegmentationModule):
    """Implementation of a joint prediction module for segmentation tasks. This module combines the predictions of a explicit segmentation model and an implicit one."""

    def __init__(self, 
                 segmentation_module: Optional[Module] = None, 
                 prior_module: Optional[Module] = None, 
                 prior_mode: PriorMode = PriorMode.PARTIAL, 
                 input_mode: InputMode = InputMode.PIXEL, 
                 prior_output_processing_mode: PriorOutputProcessingMode = PriorOutputProcessingMode.SIGMOID, 
                 segmentation_output_processing_mode: SegmentationOutputProcessingMode = SegmentationOutputProcessingMode.SIGMOID, 
                 gradient_mode: GradientMode = GradientMode.BOTH, 
                 evaluation_mode: EvaluationMode = EvaluationMode.BOTH, **kwargs) -> None:
        super().__init__(segmentation_module, prior_module, prior_mode, input_mode, prior_output_processing_mode, segmentation_output_processing_mode, gradient_mode, evaluation_mode, **kwargs)


    