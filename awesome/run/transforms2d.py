from decimal import Decimal
import torch
from typing import Literal, Optional, Tuple, TypeVar, Union
from awesome.util.torch import tensorify
import numpy as np

import numpy as np


VEC_TYPE = TypeVar("VEC_TYPE", bound=Union[torch.Tensor, np.ndarray])
"""Vector type, like torch.Tensor or numpy.ndarray."""

NUMERICAL_TYPE = TypeVar("NUMERICAL_TYPE", bound=Union[torch.Tensor, np.generic, int, float, complex, Decimal])
"""Numerical type which can be converted to a tensor."""


__all__ = [
    "assure_affine_vector",
    "assure_affine_matrix",
    "unit_vector",
    "component_rotation_matrix",
    "component_position_matrix",
    "component_transformation_matrix",
    "transformation_matrix",
    "scale_matrix",
    "split_transformation_matrix",
    "vector_angle"
]


def assure_affine_vector(_input: VEC_TYPE,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         requires_grad: bool = False) -> torch.Tensor:
    """Assuring the _input vector instance is an affine vector.
    Converting it into tensor if nessesary.
    Adds 1 to the vector if its size is 2.

    Parameters
    ----------
    _input : Union[torch.Tensor, np.ndarray]
        Vector of length 2 or 3.
    dtype : Optional[torch.dtype], optional
        The dtype of the tensor, by default None
    device : Optional[torch.device], optional
        Its device, by default None
    requires_grad : bool, optional
        Whether it requires grad and the input was numpy array, by default False

    Returns
    -------
    torch.Tensor
        The affine tensor.

    Raises
    ------
    ValueError
        If shape is wrong.
    """
    _input = tensorify(_input, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    if len(_input.shape) != 1:
        raise ValueError(f"assure_affine_vector works only on 1 d tensors!")
    if _input.shape[0] > 3 or _input.shape[0] < 2:
        raise ValueError(
            f"assure_affine_vector works only for tensors of length 2 or 3.")
    if _input.shape[0] == 3:
        return _input  # Assuming it contains already affine property
    else:
        # Length of 2
        return torch.cat([_input, torch.tensor([1], device=_input.device, 
                                               dtype=_input.dtype, 
                                               requires_grad=_input.requires_grad)])


def assure_affine_matrix(_input: VEC_TYPE,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         requires_grad: bool = False) -> torch.Tensor:
    """Assuring the _input matrix instance is an affine matrix.
    Converting it into tensor if nessesary.

    Parameters
    ----------
    _input : Union[torch.Tensor, np.ndarray]
        Matrix of x / y shape 2 or 3.
    dtype : Optional[torch.dtype], optional
        The dtype of the tensor, by default None
    device : Optional[torch.device], optional
        Its device, by default None
    requires_grad : bool, optional
        Whether it requires grad and the input was numpy array, by default False

    Returns
    -------
    torch.Tensor
        The affine tensor.

    Raises
    ------
    ValueError
        If shape is wrong.
    """
    _input = tensorify(_input, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    if len(_input.shape) != 2:
        raise ValueError(f"assure_affine_matrix works only on 2 d tensors!")
    if _input.shape[0] > 3 or _input.shape[0] < 2:
        raise ValueError(
            f"assure_affine_matrix works only for tensors of length 2 or 3.")
    if _input.shape[0] == 3:
        pass
    else:
        # Length of 3
        _input = torch.cat(
            [_input, torch.tensor(
                [[0., 0.] + ([] if _input.shape[1] == 2 else [1.])],
                device=_input.device, dtype=_input.dtype, requires_grad=_input.requires_grad)],
            axis=-2)
    if _input.shape[1] > 3 or _input.shape[1] < 2:
        raise ValueError(
            f"assure_affine_matrix works only for tensors of length 2 or 3.")
    if _input.shape[1] == 3:
        pass
    else:
        # Length of 3
        _input = torch.cat([_input, torch.tensor(
            [[0., 0., 1.]], device=_input.device, dtype=_input.dtype, requires_grad=_input.requires_grad).T], axis=-1)
    return _input


def is_transformation_matrix(_input: VEC_TYPE) -> torch.Tensor:
    """Returns whether a given input is a numpy or torch
    matrix of size (2 x 2) / (3 x 3) which can be used as transformation matricies.

    Parameters
    ----------
    _input : VEC_TYPE
        The input to check.

    Returns
    -------
    torch.Tensor
        Whether the input is a transformation matrix.
    """
    if _input is None:
        return False
    if isinstance(_input, (torch.Tensor, np.ndarray)):
        if tuple(_input.shape) in [(2, 2), (3, 3)]:
            return True
    return False


def is_position_vector(_input: VEC_TYPE) -> torch.Tensor:
    """Returns whether a given input is a numpy or torch
    matrix of size (3, ) / (4, ) which can be used as position vector.

    Parameters
    ----------
    _input : VEC_TYPE
        The input to check.

    Returns
    -------
    torch.Tensor
        Whether the input is a transformation vector.
    """
    if _input is None:
        return False
    if isinstance(_input, (torch.Tensor, np.ndarray)):
        if tuple(_input.shape) in [(2,), (3, )]:
            return True
    return False

def split_transformation_matrix(_input: VEC_TYPE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a transformation matrix in its position and orientation component.

    Parameters
    ----------
    _input : VEC_TYPE
        The input matrix

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The position (3, ) and orientation matrix (3, 3).

    Raises
    ------
    ValueError
        If input shape is of invalid shape.
    """
    if tuple(_input.shape) != (3, 3):
        raise ValueError(f"Invalid shape for split: {_input.shape}")
    position = _input[:2, 2]
    orientation = _input[:2, :2]
    return position, orientation


def unit_vector(_input: torch.Tensor) -> torch.Tensor:
    """Calculates a unit vector out of the input.

    Parameters
    ----------
    _input : torch.Tensor
        The input.

    Returns
    -------
    torch.Tensor
        The normed input vector (unit-vector).
    """
    return _input / torch.norm(_input)


def component_rotation_matrix(angle: Optional[NUMERICAL_TYPE] = None,
                              mode: Literal["deg", "rad"] = 'rad',
                              dtype: torch.dtype = None,
                              device: torch.device = None,
                              requires_grad: bool = False) -> torch.Tensor:
    """Computes the rotation matrix out of angles.

    Parameters
    ----------
    angle : Optional[NUMERICAL_TYPE], optional
        Rotation angle, by default None
    mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
        If the angles are specified in radians [0, 2*pi), or degrees [0, 360), by default 'rad'
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default None
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device.
    Returns
    -------
    torch.Tensor
        The rotation matrix of shape (3, 3)
    """
    angle = tensorify(
        angle, dtype=dtype, device=device, requires_grad=requires_grad) if angle is not None else torch.tensor(
        0, dtype=dtype, device=device, requires_grad=requires_grad)

    if mode == "deg":
        angle = torch.deg2rad(angle)

    rot = torch.tensor(np.identity(3), dtype=dtype, device=device, requires_grad=requires_grad)
    if dtype is None:
        rot = rot.to(dtype=torch.float32)  # Default dtype for torch.tensor
    if angle != 0.:
        r_x = torch.zeros((3, 3), dtype=rot.dtype, device=rot.device, requires_grad=rot.requires_grad)
        r_x[0, 0] = 1
        r_x[1, 1] = torch.cos(angle)
        r_x[1, 2] = -torch.sin(angle)
        r_x[2, 1] = torch.sin(angle)
        r_x[2, 2] = torch.cos(angle)
        r_x[3, 3] = 1
        return r_x
    return rot


def component_transformation_matrix(x: Optional[NUMERICAL_TYPE] = None,
                                    y: Optional[NUMERICAL_TYPE] = None,
                                    dtype: torch.dtype = None,
                                    device: torch.device = None,
                                    requires_grad: bool = False) -> torch.Tensor:
    """Returng a transformation matrix based on given components.

    Parameters
    ----------
    x : Optional[NUMERICAL_TYPE], optional
        X component for the transformation, by default None
    y : Optional[NUMERICAL_TYPE], optional
        Y component for the transformation, by default None
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False 
    Returns
    -------
    torch.Tensor
        Transformation matrix.
    """
    mat = torch.tensor(np.identity(3), dtype=dtype, device=device, requires_grad=requires_grad)
    if dtype is None:
        mat = mat.to(dtype=torch.float32) # Default dtype for torch.tensor
    mat[0, 2] = x if x is not None else 0.
    mat[1, 2] = y if y is not None else 0.
    return mat

def component_scale_matrix(x: Optional[NUMERICAL_TYPE] = None,
                                    y: Optional[NUMERICAL_TYPE] = None,
                                    dtype: torch.dtype = None,
                                    device: torch.device = None,
                                    requires_grad: bool = False) -> torch.Tensor:
    """Returng a scale matrix based on given components.

    Parameters
    ----------
    x : Optional[NUMERICAL_TYPE], optional
        X component for the transformation, by default None
    y : Optional[NUMERICAL_TYPE], optional
        Y component for the transformation, by default None
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False 
    Returns
    -------
    torch.Tensor
        Transformation matrix.
    """
    mat = torch.tensor(np.identity(3), dtype=dtype, device=device, requires_grad=requires_grad)
    if dtype is None:
        mat = mat.to(dtype=torch.float32) # Default dtype for torch.tensor
    mat[0, 0] = x if x is not None else 0.
    mat[1, 1] = y if y is not None else 0.
    return mat

def transformation_matrix(vector: VEC_TYPE,
                          dtype: torch.dtype = None,
                          device: torch.device = None,
                          requires_grad: bool = False) -> torch.Tensor:
    """Getting the transformation matrix from a vector.

    Parameters
    ----------
    vector : VEC_TYPE
        The vector to transform for.
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False 

    Returns
    -------
    torch.Tensor
        The resulting transformation matrix.
    """
    vector = tensorify(vector, dtype=dtype, device=device, requires_grad=requires_grad)
    mat = torch.tensor(np.identity(3), dtype=dtype, device=device, requires_grad=requires_grad)
    if dtype is None:
        mat = mat.to(dtype=torch.float32) # Default dtype for torch.tensor
    mat[0:2, 2] = vector[0:2]
    return mat


def scale_matrix(vector: VEC_TYPE,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 requires_grad: bool = False) -> torch.Tensor:
    """Getting the scale matrix from a vector.

    Parameters
    ----------
    vector : VEC_TYPE
        The vector to get scale from.
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False 

    Returns
    -------
    torch.Tensor
        The resulting scale matrix.
    """
    vector = tensorify(vector, dtype=dtype, device=device, requires_grad=requires_grad)
    mat = torch.tensor(np.identity(3), dtype=dtype, device=device, requires_grad=requires_grad)
    mat[0, 0] = vector[0]
    mat[1, 1] = vector[1]
    return mat



def vector_angle(v1: VEC_TYPE, v2: VEC_TYPE) -> torch.Tensor:
    """Computes the angle between vector v1 and v2.

    Parameters
    ----------
    v1 : VEC_TYPE
        The first input vector
    v2 : VEC_TYPE
        The second input vector

    Returns
    -------
    torch.Tensor
        Angle between vector.
    """
    return torch.acos(torch.dot(v1, v2) / (torch.norm(v1, keepdim=True) * torch.norm(v2, keepdim=True)))