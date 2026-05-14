"""
Tensor construction and manipulation utilities for KinoPulse.

Provides utilities for creating, batching, and manipulating tensors
with consistent conventions across the library.
"""

from typing import Any, Optional, Union
import torch

# Numpy is optional - only used for conversion in ensure_tensor
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .device import get_default_device
from .dtype import get_default_dtype


def zeros_like_state(
    state: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Create a zero tensor with the same properties as a state.

    Args:
        state: Reference state tensor
        batch_size: Optional batch size (overrides state's batch size if provided)

    Returns:
        Zero tensor with matching shape, device, and dtype

    Example:
        >>> state = torch.randn(4, 10)  # batch_size=4, state_dim=10
        >>> zeros = zeros_like_state(state)
        >>> zeros.shape
        torch.Size([4, 10])
        >>> zeros = zeros_like_state(state, batch_size=8)
        >>> zeros.shape
        torch.Size([8, 10])
    """
    if batch_size is not None:
        # Replace batch dimension
        shape = (batch_size,) + state.shape[1:]
        return torch.zeros(shape, device=state.device, dtype=state.dtype)
    else:
        return torch.zeros_like(state)


def ones_like_state(
    state: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Create a ones tensor with the same properties as a state.

    Args:
        state: Reference state tensor
        batch_size: Optional batch size (overrides state's batch size if provided)

    Returns:
        Ones tensor with matching shape, device, and dtype

    Example:
        >>> state = torch.randn(4, 10)
        >>> ones = ones_like_state(state)
        >>> ones.shape
        torch.Size([4, 10])
    """
    if batch_size is not None:
        shape = (batch_size,) + state.shape[1:]
        return torch.ones(shape, device=state.device, dtype=state.dtype)
    else:
        return torch.ones_like(state)


def eye_like(
    dim: int,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create batched identity matrices.

    Args:
        dim: Matrix dimension (creates dim x dim identity)
        batch_size: Optional batch size (creates batch of identities)
        device: Device to place tensor on (uses default if None)
        dtype: Dtype for tensor (uses default if None)

    Returns:
        Identity matrix or batch of identities

    Example:
        >>> I = eye_like(3)
        >>> I.shape
        torch.Size([3, 3])
        >>> I_batch = eye_like(3, batch_size=4)
        >>> I_batch.shape
        torch.Size([4, 3, 3])
    """
    if device is None:
        device = get_default_device()
    if dtype is None:
        dtype = get_default_dtype()

    eye = torch.eye(dim, device=device, dtype=dtype)

    if batch_size is not None:
        # Expand to batch
        eye = eye.unsqueeze(0).expand(batch_size, -1, -1)

    return eye


def batch_over(
    tensor: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Broadcast a tensor over a batch dimension.

    If tensor has no batch dimension, unsqueeze and expand.
    If tensor already has batch dimension, check compatibility.

    Args:
        tensor: Input tensor
        batch_size: Target batch size

    Returns:
        Tensor with batch dimension

    Example:
        >>> x = torch.randn(10)  # no batch
        >>> x_batch = batch_over(x, batch_size=4)
        >>> x_batch.shape
        torch.Size([4, 10])
    """
    if tensor.ndim == 0:
        # Scalar tensor, expand to batch
        return tensor.unsqueeze(0).expand(batch_size)
    elif tensor.shape[0] == batch_size:
        # Already has correct batch size
        return tensor
    elif tensor.shape[0] == 1:
        # Has batch dim of 1, expand
        shape = (batch_size,) + tensor.shape[1:]
        return tensor.expand(*shape)
    else:
        # Assume no batch dim, add one
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)


def clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Clone a tensor preserving requires_grad flag.

    Args:
        tensor: Input tensor

    Returns:
        Cloned tensor with same requires_grad setting

    Example:
        >>> x = torch.randn(10, requires_grad=True)
        >>> y = clone_with_grad(x)
        >>> y.requires_grad
        True
    """
    cloned = tensor.clone()
    if tensor.requires_grad:
        cloned.requires_grad_(True)
    return cloned


def safe_detach(tensor: torch.Tensor) -> torch.Tensor:
    """Safely detach a tensor from the computation graph.

    Args:
        tensor: Input tensor

    Returns:
        Detached tensor

    Example:
        >>> x = torch.randn(10, requires_grad=True)
        >>> y = safe_detach(x)
        >>> y.requires_grad
        False
    """
    return tensor.detach()


def ensure_tensor(
    obj: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert various inputs to torch.Tensor.

    Handles numpy arrays, lists, scalars, and existing tensors.

    Args:
        obj: Input to convert (tensor, numpy array, list, scalar, etc.)
        dtype: Optional target dtype (uses default if None)
        device: Optional target device (uses default if None)

    Returns:
        Torch tensor

    Example:
        >>> x = ensure_tensor([1, 2, 3])
        >>> x
        tensor([1., 2., 3.])
        >>> x = ensure_tensor(np.array([1, 2, 3]))
        >>> x
        tensor([1., 2., 3.])
    """
    if device is None:
        device = get_default_device()
    if dtype is None:
        dtype = get_default_dtype()

    # Already a tensor
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype)

    # Numpy array (if numpy is available)
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        tensor = torch.from_numpy(obj).to(device=device)
        # Convert to target dtype if needed
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    # List or tuple
    if isinstance(obj, (list, tuple)):
        tensor = torch.tensor(obj, device=device, dtype=dtype)
        return tensor

    # Scalar (int, float)
    if isinstance(obj, (int, float)):
        tensor = torch.tensor(obj, device=device, dtype=dtype)
        return tensor

    # Unsupported type
    raise TypeError(f"Cannot convert {type(obj)} to torch.Tensor")
