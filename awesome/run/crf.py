
from dataclasses import dataclass, field
from typing import Any, Tuple
import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from awesome.util.torch import VEC_TYPE

@dataclass
class CRFOptions():
    """Options for the CRF."""

    max_iterations: int = 15
    """Iterations for the CRF inference."""

    gaussian_sdims: Tuple[float, float] = (3, 3)
    """Scaling factors for the gaussian kernel. Refer to `sxy` in `DenseCRF2D.addPairwiseGaussian`."""

    gaussian_compat: float = 3
    """Compatibility for the gaussian kernel. Refer to `compat` in `DenseCRF2D.addPairwiseGaussian`."""

    gaussian_kernel: Any = dcrf.DIAG_KERNEL
    """Kernel for the gaussian kernel. Refer to `kernel` in `DenseCRF2D.addPairwiseGaussian`."""

    gaussian_normalization: Any = dcrf.NORMALIZE_SYMMETRIC
    """Normalization for the gaussian kernel. Refer to `normalization` in `DenseCRF2D.addPairwiseGaussian`."""

    bilateral_sdims: Tuple[float, float] = (50, 50)
    """Scaling factors for the bilateral kernel. Refer to `sxy` in `DenseCRF2D.addPairwiseBilateral`."""

    bilateral_schan: Tuple[float, float, float] = (10, 10, 10)
    """Scaling factors for the bilateral kernel. Refer to `srgb` in `DenseCRF2D.addPairwiseBilateral`."""

    bilateral_compat: float = 5
    """Compatibility for the bilateral kernel. Refer to `compat` in `DenseCRF2D.addPairwiseBilateral`."""

    bilateral_kernel: Any = dcrf.DIAG_KERNEL
    """Kernel for the bilateral kernel. Refer to `kernel` in `DenseCRF2D.addPairwiseBilateral`."""

    bilateral_normalization: Any = dcrf.NORMALIZE_SYMMETRIC
    """Normalization for the bilateral kernel. Refer to `normalization` in `DenseCRF2D.addPairwiseBilateral`."""


def dense_crf(img: VEC_TYPE, unaries: VEC_TYPE, 
              is_softmax_unaries: bool = True, 
              options: CRFOptions = None) -> np.ndarray:
    """
    Creates a dense CRF for the given image and unaries.

    Parameters
    ----------
    img: numpy.array or torch.Tensor
        The input image. If torch.Tensor, it should have shape (c, h, w) or (h, w).
        If numpy.array, it should have shape (h, w, c) or (h, w).
    unaries: numpy.array or torch.Tensor
        The unaries for the CRF. If torch.Tensor, it should have shape (c, h, w) or (h, w).
        If numpy.array, it should have shape (h, w, c) or (h, w).
        If the unaries are softmaxed, the unaries should be in the range [0, 1].
        If the channel dimension is 1, the unaries are assumed to be for a binary cross entropy.
        It will create 2 channels for the CRF, splitting the binary cross entropy in a cross entropy like format,
        where the first class is the likelihood and the second 1 - likelihood.
    is_softmax_unaries: bool, optional
        Whether the unaries are softmaxed or not. E.g. sum to 1. If True, the unaries
        are converted to unary potentials.
    options: CRFOptions, optional
        The options for the CRF. If None, the default options are used.

    Returns
    -------
    numpy.array
        The CRF output. Has shape (h, w, c).
    """

    if isinstance(unaries, torch.Tensor):
        if len(unaries.shape) == 3:
            unaries = unaries.permute(1, 2, 0)
        unaries = unaries.detach().cpu().numpy()
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()

    if len(unaries.shape) == 2:
        unaries = unaries[:, :, None]

    if len(img.shape) == 2:
        img = img[:, :, None]

    if options is None:
        options = CRFOptions()

    unaries = unaries.transpose(2, 0, 1)
    
    c = unaries.shape[0]
    h = unaries.shape[1]
    w = unaries.shape[2]

    labels = np.zeros((2, img.shape[0], img.shape[1]))

    single_channel = False

    if c == 1:
        labels[0, :, :] = unaries[0]
        labels[1, :, :] = 1 - unaries[0]
        c = 2
        single_channel = True
    else:
        labels = unaries

    if is_softmax_unaries:
        U = unary_from_softmax(labels)

    U = np.ascontiguousarray(U)

    # Image should have dim (h, w, c)
    if img.dtype != np.uint8:
        img = img * 255.0
        img = img.astype(np.uint8)

    img = np.ascontiguousarray(img)

    crf = dcrf.DenseCRF2D(w, h, c)
    crf.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=options.gaussian_sdims, shape=img.shape[:2])
    crf.addPairwiseEnergy(feats, 
                          compat=options.gaussian_compat, 
                          kernel=options.gaussian_kernel,
                          normalization=options.gaussian_normalization)
    
    feats = create_pairwise_bilateral(sdims=options.bilateral_sdims, schan=options.bilateral_schan, img=img, chdim=2)
    crf.addPairwiseEnergy(feats, 
                          compat=options.bilateral_compat, 
                          kernel=options.bilateral_kernel,
                          normalization=options.bilateral_normalization)
        
    Q = crf.inference(options.max_iterations)
    Q = np.array(Q).reshape((c, h, w))

    if single_channel:
        # Selecting the channel which is not 1 - value
        Q = Q[0][None, ...]

    Q = Q.transpose(1, 2, 0)
    return Q
