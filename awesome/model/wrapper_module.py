import copy
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
from awesome.model.abstract_combined_segmentation_module import AbstractCombinedSegmentationModule, EvaluationMode, InputMode, PriorMode
from awesome.model.pretrainable_module import PretrainableModule
from awesome.util.batcherize import batcherize
from awesome.util.prior_cache import PriorCache
from awesome.util.torch import TensorUtil


class WrapperModule(AbstractCombinedSegmentationModule, PretrainableModule):
    """legacy model for learning."""

    def __init__(self,
                 segmentation_module: nn.Module = None,
                 prior_module: Optional[nn.Module] = None,
                 mode: str = "single",  # Refer as segmentation training mode
                 full_prior: bool = False,
                 segmentation_arg_mode: Literal['forward'] = 'forward',
                 prior_arg_mode: Literal['none',
                                         'xy_c_preattached',
                                         'param_grid',
                                         'param_clean_grid'
                                         ] = 'xy_c_preattached',
                 input_mode: Literal['pixel', 'image'] = 'pixel',
                 use_segmentation_sigmoid: bool = True,
                 use_segmentation_output_inversion: bool = False,
                 use_prior_sigmoid: bool = True,
                 segmentation_module_gets_targets: bool = False,
                 output_mode: Literal['tensor'] = 'tensor',
                 **kwargs
                 ):
        if 'prior_mode' in kwargs:
            kwargs.pop('prior_mode')
        super().__init__(segmentation_module=segmentation_module,
                         prior_module=prior_module,
                         input_mode=input_mode,
                         prior_mode=PriorMode.PARTIAL if not full_prior else PriorMode.FULL,
                         **kwargs)
        self.mode = mode
        self.prior_arg_mode = prior_arg_mode
        self.segmentation_arg_mode = segmentation_arg_mode
        self.use_segmentation_sigmoid = use_segmentation_sigmoid
        self.use_segmentation_output_inversion = use_segmentation_output_inversion
        self.segmentation_module_gets_targets = segmentation_module_gets_targets
        self.use_prior_sigmoid = use_prior_sigmoid
        self.output_mode = output_mode

    @property
    def evaluate_prior(self) -> bool:
        """Legacy property for evaluating the prior module.

        This is deprecated and should not be used anymore.

        Returns
        -------
        bool
            If the prior module should be evaluated.
        """
        return self.evaluation_mode == EvaluationMode.BOTH or self.evaluation_mode == EvaluationMode.PRIOR

    @evaluate_prior.setter
    def evaluate_prior(self, value: bool) -> None:
        """Legacy property for evaluating the prior module.

        This is deprecated and should not be used anymore.

        Parameters
        ----------
        value : bool
            If the prior module should be evaluated.
        """
        if value:
            self.evaluation_mode = EvaluationMode.BOTH
        else:
            self.evaluation_mode = EvaluationMode.SEGMENTATION

    def get_prior_args(self,
                       _input: torch.Tensor,
                       *args,
                       segm: Optional[Any] = None,
                       **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        
        prior_args = list()
        prior_kwargs = dict()
        input_mode = self.input_mode
        if isinstance(input_mode, InputMode):
            input_mode = input_mode.value
        if input_mode == 'pixel':
            # Input mode (B, C)
            if self.prior_arg_mode == 'none':
                return prior_args, prior_kwargs
            elif self.prior_arg_mode == 'xy_c_preattached':
                # Assuming that input is of shape (B, C)
                prior_args.append(_input[..., 0:2])
                return prior_args, prior_kwargs
            elif self.prior_arg_mode == 'param_clean_grid':
                # Assuming there is an additional parameter grid and clean_grid
                prior_args.append(args[1])
                return prior_args, prior_kwargs
            else:
                raise NotImplementedError(
                    f"prior_args must be either 'none' or 'xy_c_preattached' but is {self.prior_arg_mode}")
        elif input_mode == 'image':
            # Input mode (B, C, H, W)
            if self.prior_arg_mode == 'none':
                return prior_args, prior_kwargs
            elif self.prior_arg_mode == 'xy_c_preattached':
                # TODO: Deprecated!
                # Assuming that input is of shape (B, C, H, W)
                prior_args.append(_input[:, 0:2])
                return prior_args, prior_kwargs
            elif self.prior_arg_mode == 'param_grid':
                # TODO: Deprecated!
                # Assuming there is an additional parameter grid
                prior_args.append(args[0])
                return prior_args, prior_kwargs
            elif self.prior_arg_mode == 'param_clean_grid':
                # Assuming there is an additional parameter grid and clean_grid
                grid: torch.Tensor = args[1]
                prior_args.append(grid)
                return prior_args, prior_kwargs
        else:
            raise ValueError(
                f"input_mode must be either 'pixel' or 'image' but is {self.input_mode}")

    def get_assure_single_batch(self, index: int) -> Any:
        def _assure_single_batch(_input: torch.Tensor) -> torch.Tensor:
            nonlocal self
            nonlocal index
            if self.input_mode == InputMode.PIXEL:
                return _input[index]
            elif self.input_mode == InputMode.IMAGE:
                return _input[index][None, ...]
            else:
                raise ValueError(
                    f"input_mode must be either 'pixel' or 'image' but is {self.input_mode}")
        return _assure_single_batch

    def get_segmentation_module_args(self, primary: Any, args: Tuple[Any], kwargs: Dict[str, Any], targets: Any) -> Tuple[Any, Tuple[Any], Dict[str, Any]]:
        # Ignore the 2nd parameter if any for segmentation module as this is xy_clean just for the prior
        new_args_segm = args[:1]
        if len(args) > 2:
            new_args_segm += args[2:]
        if self.segmentation_module_gets_targets:
            kwargs = dict(kwargs)
            kwargs['targets'] = targets

        if self.segmentation_arg_mode == 'forward':
            return primary, new_args_segm, kwargs
        else:
            raise NotImplementedError(
                f"segmentation_arg_mode {self.segmentation_arg_mode} is unknown.")

    @batcherize(expected_dim=3)  # Prep to at least 3 dimensions
    def forward(self,
                _input: torch.Tensor,
                *args,
                targets: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Gets the pixels of an image and returns the segmentation and convexity output of the pixels.

        Parameters
        ----------
        _input : torch.Tensor
            Pixel which should be segmented and checked for convexity.
            Shape: (img, n_pixels, 5) where img is the number of images, n_pixels is the number of pixels in the image and 5 is the YX Coordinate + RGB value of the pixel.

            OR

            (B, C, H, W) where B is the batch size, C is the number of channels, H is the height and W is the width of the image.
            B should typicallly be 1 so priors can work properly.

        *args : Any
            Additional arguments for the segmentation and prior module.

        **kwargs : Any
            Additional keyword arguments for the segmentation and prior module.
        Returns
        -------
        torch.Tensor
            Segmentation and convexity output of the pixels.
            Shape: (img, n_pixels, 2) where img is the number of images, n_pixels is the number of pixels in the image and 2 is the segmentation and convexity output of the pixel.

            OR

            (B, C, H, W) where B is the batch size, C is the number of channels, H is the height and W is the width of the image.
        """

        # The last dimension is the YX Coordinate + RGB value of the pixel.
        # For each image, extract the scribbled pixels
        img_res = []

        for i in range(_input.shape[0]):
            fnc = self.get_assure_single_batch(i)
            # Batches are seperate images in this case
            _single_image_input = fnc(_input)
            new_args: tuple = TensorUtil.apply_deep(args, fnc=fnc)
            new_kwargs: tuple = TensorUtil.apply_deep(kwargs, fnc=fnc)
            new_targets = TensorUtil.apply_deep(targets, fnc=fnc)
            seg_input, seg_input_args, seg_input_kwargs = self.get_segmentation_module_args(
                _single_image_input, new_args, new_kwargs, targets=new_targets)

            with torch.set_grad_enabled(self.segmentation_module.training != "none"):
                segm_out = self.segmentation_module(
                    seg_input, *seg_input_args, **seg_input_kwargs)
            segm_out = self.process_segmentation_output(segm_out)

            prior_out = None

            if self.prior_module is not None and self.evaluate_prior:
                # Args for the prior:
                prior_args, prior_kwargs = self.get_prior_args(_single_image_input,
                                                               *new_args,
                                                               segm=segm_out,
                                                               **new_kwargs
                                                               )
                # Convexity gets YX-Coordinate
                prior_out = self.prior_module(*prior_args, **prior_kwargs)
                prior_out = self.process_prior_output(prior_out)

            self.combine_outputs(segm_out, prior_out, img_res)

        if self.output_mode == 'tensor':
            img_res = torch.stack(img_res, dim=0)
        return img_res

    def combine_outputs(self,
                        segm: Any,
                        prior: Any,
                        stack: List[Any]
                        ) -> None:
        if self.output_mode == 'tensor':
            if self.prior_module is not None and self.evaluate_prior:
                if self.input_mode == 'pixel':
                    stack.append(torch.cat([segm, prior], dim=-1))
                else:
                    stack.append(torch.cat([segm, prior], dim=0))
            else:
                stack.append(segm)
        else:
            raise NotImplementedError(
                f"output_mode must be either 'tensor' or 'dict' but is {self.output_mode}")

    def process_segmentation_output(self, segm: Any) -> Any:
        if self.output_mode == 'tensor':
            # if segm has batch dimension of one, remove it, will be added on stack
            if segm.shape[0] == 1:
                segm = segm[0]

            if self.use_segmentation_sigmoid:
                # Sigmoid to get values between 0 and 1
                segm_sig = torch.sigmoid(segm)
            else:
                segm_sig = segm
            if self.use_segmentation_output_inversion:
                segm_sig = 1 - segm_sig
            return segm_sig
        else:
            raise NotImplementedError(
                f"output_mode must be either 'tensor' or 'dict' but is {self.output_mode}")

    def process_prior_output(self, prior: Any, use_sigmoid: Optional[bool] = None, squeeze: bool = True) -> Any:
        if isinstance(prior, torch.Tensor) and squeeze:
            if prior.shape[0] == 1 and len(prior.shape) == 4:
                prior = prior[0]
        if (use_sigmoid is None and self.use_prior_sigmoid) or use_sigmoid:
            prior_sig = TensorUtil.apply_deep(prior, torch.sigmoid)
        else:
            prior_sig = prior
        return prior_sig

    @batcherize(keep=True, expected_dim=3)
    def split_model_output(self, output: Any, additional_data: Optional[Dict[str, Any]] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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
        input_mode = self.input_mode
        if isinstance(input_mode, InputMode):
            input_mode = input_mode.value
        if self.output_mode == 'tensor':
            for batch_idx in range(output.shape[0]):
                batch_output = output[batch_idx]
                seg = None
                conv = None
                if input_mode == 'pixel':
                    # Assuming N x C for pixel wise 2d Data
                    if self.prior_module is not None:
                        seg = batch_output[..., :batch_output.shape[-1] // 2]
                        conv = batch_output[..., batch_output.shape[-1] // 2:]
                    else:
                        seg = batch_output
                elif input_mode == 'image':
                    # Assuming C x H x W for image wise 3d Data
                    if self.prior_module is not None:
                        seg = batch_output[:batch_output.shape[0] // 2, ...]
                        conv = batch_output[batch_output.shape[0] // 2:, ...]
                    else:
                        seg = batch_output
                else:
                    raise ValueError(
                        f"input_mode must be either 'pixel' or 'image' but is {input_mode}")
                res.append((seg, conv))
        else:
            raise NotImplementedError(
                f"output_mode {self.output_mode} is unknown.")
        return res

    def enforce_convexity(self) -> None:
        if self.prior_module is not None:
            self.prior_module.enforce_convexity()

    def pretrain(self,
                 *args,
                 **kwargs) -> Any:
        if not isinstance(self.prior_module, PretrainableModule):
            raise ValueError("Prior module must be a PretrainableModule")
        return self.prior_module.pretrain(*args,
                                          wrapper_module=self,
                                          **kwargs)

    def pretrain_load_state(self, *args, **kwargs) -> None:
        if not isinstance(self.prior_module, PretrainableModule):
            raise ValueError("Prior module must be a PretrainableModule")
        self.prior_module.pretrain_load_state(*args,
                                              wrapper_module=self,
                                              **kwargs
                                              )
