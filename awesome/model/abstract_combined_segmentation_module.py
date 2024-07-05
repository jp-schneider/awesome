from abc import abstractmethod
import copy
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, TYPE_CHECKING
import torch
from torch.utils.data.dataset import Dataset
from awesome.model.pretrainable_module import PretrainableModule
from awesome.util.batcherize import batcherize
from awesome.util.prior_cache import PriorCache
from awesome.util.torch import TensorUtil
from awesome.util.format import parse_enum

class PriorMode(Enum):
    FULL = "full"
    """If the prior mode is full, the prior is extracted and applied to the module as a whole. Also the segmentation module is a prior and weights are swapped accordingly."""
    
    PARTIAL = "partial"
    """If the prior mode is partial, the prior module is a prior and will be changed while the segmentation module stays fixed."""

    NONE = "none"
    """If the prior mode is none, no prior is used at all => The weights of all model parts stay the same accross the evaluated data."""

class InputMode(Enum):

    PIXEL = "pixel"
    """If the input mode is pixel, the input is assumed to be a pixel-wise 3D tensor (B, H x W, C)."""
    
    IMAGE = "image"
    """If the input mode is image, the input is assumed to be an image-wise 4D tensor (B, C, H, W)."""

class PriorOutputProcessingMode(Enum):

    NONE = "none"
    """If the prior output processing mode is none, the prior output is not processed at all."""
    
    SIGMOID = "sigmoid"
    """If the prior output processing mode is sigmoid, the prior output is processed with a sigmoid function."""

class SegmentationOutputProcessingMode(Enum):
    
    NONE = "none"
    """If the segmentation output processing mode is none, the segmentation output is not processed at all."""
    
    SIGMOID = "sigmoid"
    """If the segmentation output processing mode is sigmoid, the segmentation output is processed with a sigmoid function."""

    SIGMOID_INVERTED = "sigmoid_inverted"
    """If the prior output processing mode is sigmoid_inverted, the prior output is processed with a sigmoid function and fg bg are inverted by 1 - x."""

class GradientMode(Enum):
    """Enum for the gradient mode. Controls which gradients are tracked during forward pass."""

    NONE = "none"
    """If the gradient mode is none, no gradients are tracked during forward pass."""
    
    SEGMENTATION = "segmentation"
    """If the gradient mode is segmentation, gradients are tracked for the segmentation module."""
    
    PRIOR = "prior"
    """If the gradient mode is prior, gradients are tracked for the prior module."""

    BOTH = "both"
    """If the gradient mode is both, gradients are tracked for both modules."""

class EvaluationMode(Enum):

    SEGMENTATION = "segmentation"
    """If the evaluation mode is segmentation, the segmentation module is evaluated."""
    
    PRIOR = "prior"
    """If the evaluation mode is prior, the prior module is evaluated."""

    BOTH = "both"
    """If the evaluation mode is both, both modules are evaluated."""

class AbstractCombinedSegmentationModule(torch.nn.Module):
    """A combined segmentation module that can be used to combine a segmentation module with a prior module."""

    segmentation_module: torch.nn.Module
    """The segmentation module that is used to segment the image. Usually a pixel-wise classifier."""

    prior_module: Optional[torch.nn.Module]
    """The prior module that is used to create one or multiple a priors for the segmentation module. Will be an implicit representation."""

    prior_mode: PriorMode
    """Prior mode to control the prior swapping behavior."""

    def __init__(self,
                segmentation_module: torch.nn.Module = None, 
                prior_module: Optional[torch.nn.Module] = None,
                prior_mode: PriorMode = PriorMode.PARTIAL,
                input_mode: InputMode = InputMode.PIXEL,
                prior_output_processing_mode: PriorOutputProcessingMode = PriorOutputProcessingMode.SIGMOID,
                segmentation_output_processing_mode: SegmentationOutputProcessingMode = SegmentationOutputProcessingMode.SIGMOID,
                gradient_mode: GradientMode = GradientMode.BOTH,
                evaluation_mode: EvaluationMode = EvaluationMode.BOTH,
                **kwargs) -> None:
            super().__init__(**kwargs)
            self.segmentation_module = segmentation_module
            self.prior_module = prior_module
            self.prior_mode = parse_enum(PriorMode, prior_mode)
            self.input_mode = parse_enum(InputMode, input_mode)
            self.prior_output_processing_mode = parse_enum(PriorOutputProcessingMode, prior_output_processing_mode)
            self.segmentation_output_processing_mode = parse_enum(SegmentationOutputProcessingMode, segmentation_output_processing_mode)
            self.gradient_mode = parse_enum(GradientMode, gradient_mode)
            self.evaluation_mode = parse_enum(EvaluationMode, evaluation_mode)
 
    def extract_prior(self) -> Mapping[str, torch.Tensor]:
        """Extracts the prior state from the prior module and returns it.

        Returns
        -------
        Mapping[str, torch.Tensor]
            The prior state, which is a mapping of parameter names to parameter values.
        """
        if self.prior_mode == PriorMode.PARTIAL:
            if self.prior_module is None:
                return None
            return PriorCache.extract_prior(self.prior_module)
        elif self.prior_mode == PriorMode.FULL:
            sharing_dict = self.state_dict()
            detached_dict = copy.deepcopy(sharing_dict)
            return detached_dict
        elif self.prior_mode == PriorMode.NONE:
            return None
        else:
            raise ValueError("Unknown prior mode: " + self.prior_mode)
    
    def apply_prior(self, prior: Mapping[str, torch.Tensor]) -> None:
        """Applies the prior state to the module.

        Parameters
        ----------
        prior : Mapping[str, torch.Tensor]
            The prior state, which is a mapping of parameter names to parameter values.
        """
        if self.prior_mode == PriorMode.PARTIAL:
            if self.prior_module is None:
                return
            PriorCache.apply_prior(self.prior_module, prior)
        elif self.prior_mode == PriorMode.FULL:
            self.load_state_dict(prior)
        elif self.prior_mode == PriorMode.NONE:
            pass
        else:
            raise ValueError("Unknown prior mode: " + self.prior_mode)
        
    @abstractmethod
    def get_segmentation_module_args(self, *args, context: Dict[str, Any], **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Returns the arguments for the segmentation module.
        Will be feeded with the parameters of the forward method.

        Parameters
        ----------
        args : Tuple
            The positional arguments of the forward method.
        
        context : Dict[str, Any]
            A context dictionary which key value pairs will be forwarded to all following methods within the forward method.

        kwargs : Dict[str, Any]
            The keyword arguments of the forward method.

        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            The arguments for the segmentation module.
            1. The positional arguments for the segmentation module.
            2. The keyword arguments for the segmentation module.
        """
        pass

    @abstractmethod
    def get_prior_module_args(self, *args, context: Dict[str, Any], **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Returns the arguments for the prior module.
        Will be feeded with the parameters of the forward method.
        
        Parameters
        ----------
        args : Tuple
            The positional arguments of the forward method.
        
        context : Dict[str, Any]
            A context dictionary which key value pairs will be forwarded to all following methods within the forward method.

        kwargs : Dict[str, Any]
            The keyword arguments of the forward method.

        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            The arguments for the prior module.
            1. The positional arguments for the prior module.
            2. The keyword arguments for the prior module.
        """
        pass



    def process_segmentation_module_output(self, output: torch.Tensor, context: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Processes the output of the segmentation module.

        Parameters
        ----------
        output : torch.Tensor
            Output of the segmentation module.
        context : Dict[str, Any]
            Context dictionary.

        Returns
        -------
        torch.Tensor
            The processed output of the segmentation module.
        """
        # if segm has batch dimension of one, remove it, will be added on stack
        if self.segmentation_output_processing_mode == SegmentationOutputProcessingMode.SIGMOID:
            output = torch.sigmoid(output) # Sigmoid to get values between 0 and 1
        elif self.segmentation_output_processing_mode == SegmentationOutputProcessingMode.SIGMOID_INVERTED:
            output = torch.sigmoid(output)
            output = 1 - output
        else:
            pass
        return output

    def process_prior_module_output(self, output: torch.Tensor, context: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Processes the output of the prior module.

        Parameters
        ----------
        output : torch.Tensor
            Output of the prior module.
        context : Dict[str, Any]
            Context dictionary.

        Returns
        -------
        torch.Tensor
            The processed output of the prior module.
        """
        if self.prior_output_processing_mode == PriorOutputProcessingMode.SIGMOID:
            output = torch.sigmoid(output)
        else:
            pass
        return output


    def forward(self, *args, **kwargs) -> Any:
        """Forward method of the module.

        Returns
        -------
        Any
            The output of the forward method.
        """
        context = dict()

        seg_output = None
        if self.evaluation_mode in [EvaluationMode.SEGMENTATION, EvaluationMode.BOTH]:
            seg_args, seg_kwargs = self.get_segmentation_module_args(*args, context=context, **kwargs)
            kwargs.update(context)
            with torch.set_grad_enabled(self.gradient_mode in [GradientMode.SEGMENTATION, GradientMode.BOTH]):
                seg_output = self.segmentation_module(*seg_args, **seg_kwargs)
                seg_output = self.process_segmentation_module_output(seg_output, context=context, **kwargs)
            kwargs.update(context)

        prior_output = None
        if self.evaluation_mode in [EvaluationMode.PRIOR, EvaluationMode.BOTH]:
            prior_args, prior_kwargs = self.get_prior_module_args(*args, context=context, **kwargs)
            kwargs.update(context)
            with torch.set_grad_enabled(self.gradient_mode in [GradientMode.PRIOR, GradientMode.BOTH]):
                prior_output = self.prior_module(*prior_args, **prior_kwargs)
                prior_output = self.process_prior_module_output(prior_output, context=context, **kwargs)
            kwargs.update(context)

        output = self.combine_outputs(segm=seg_output, prior=prior_output, **kwargs)
        return output

    
    def combine_outputs(self, 
                        segm: Any, 
                        prior: Any,
                        **kwargs) -> Any:
        """Combines the segmentation and prior output and returns the combined output.

        Parameters
        ----------
        segm : Any
            Segmentation output. In shape (B, H x W, C) for pixel-wise input mode and (B, C, H, W) for image-wise input mode.
        prior : Any
            Prior output. In shape (B, H x W, C) for pixel-wise input mode and (B, C, H, W) for image-wise input mode.

        Returns
        -------
        Any
            The combined output.
        """
        if self.evaluation_mode == EvaluationMode.SEGMENTATION:
            return segm
        elif self.evaluation_mode == EvaluationMode.PRIOR:
            return prior
        elif self.evaluation_mode == EvaluationMode.BOTH:
            if self.input_mode == self.input_mode.PIXEL:
                return torch.cat([segm, prior], dim=-1)
            elif self.input_mode == self.input_mode.IMAGE:
                return torch.cat([segm, prior], dim=0)
            else:
                raise ValueError("Unknown input mode: " + self.input_mode)
        else:
            raise ValueError("Unknown evaluation mode: " + self.evaluation_mode)
    
    @batcherize(keep=True, expected_dim=3)
    def split_model_output(self, output: torch.Tensor, *args, **kwargs) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Splits the output of the model into the segmentation and convexity output.

        Parameters
        ----------
        output : torch.Tensor
            Output of the model.

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor]]
            List of tuples containing the segmentation and convexity output of the model for each batch image
        """        
        res = []
        for batch_idx in range(output.shape[0]):
            batch_output = output[batch_idx]
            seg = None
            conv = None
            if self.input_mode == InputMode.PIXEL:
                # Assuming N x C for pixel wise 2d Data
                if self.prior_module is not None:
                    seg = batch_output[..., :batch_output.shape[-1] // 2]
                    conv = batch_output[..., batch_output.shape[-1] // 2:]
                else:
                    seg = batch_output
            elif self.input_mode == InputMode.IMAGE:
                # Assuming C x H x W for image wise 3d Data
                if self.prior_module is not None:
                    seg = batch_output[:batch_output.shape[0] // 2, ...]
                    conv = batch_output[batch_output.shape[0] // 2:, ...]
                else:
                    seg = batch_output
            else:
                raise ValueError("Unknown input mode: " + self.input_mode)
            res.append((seg, conv))
        
        return res