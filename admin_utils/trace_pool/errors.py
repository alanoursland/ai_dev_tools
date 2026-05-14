"""
Error types and exception classes for KinoPulse.

This module defines a hierarchy of exceptions used throughout the library,
providing clear, context-rich error messages for debugging.
"""

from typing import Any, Optional


class KinoPulseError(Exception):
    """Base exception class for all KinoPulse errors."""
    pass


class ShapeError(KinoPulseError):
    """Raised when tensor shapes don't match expected dimensions.

    Args:
        message: Human-readable error description
        expected: Expected shape(s)
        actual: Actual shape encountered
        name: Name of the tensor/object for context

    Example:
        >>> raise ShapeError(
        ...     "State shape mismatch",
        ...     expected=(batch_size, 10),
        ...     actual=x.shape,
        ...     name="state"
        ... )
    """

    def __init__(
        self,
        message: str,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        name: Optional[str] = None,
    ):
        self.expected = expected
        self.actual = actual
        self.name = name

        full_message = message
        if name:
            full_message = f"{message} (tensor: {name})"
        if expected is not None and actual is not None:
            full_message += f"\n  Expected: {expected}\n  Actual: {actual}"

        super().__init__(full_message)


class DeviceMismatchError(KinoPulseError):
    """Raised when tensors are on incompatible devices.

    Args:
        message: Human-readable error description
        devices: List of devices encountered
        names: Names of objects for context

    Example:
        >>> raise DeviceMismatchError(
        ...     "Cannot operate on tensors from different devices",
        ...     devices=[torch.device('cpu'), torch.device('cuda:0')],
        ...     names=['state', 'input']
        ... )
    """

    def __init__(
        self,
        message: str,
        devices: Optional[list] = None,
        names: Optional[list] = None,
    ):
        self.devices = devices
        self.names = names

        full_message = message
        if devices:
            full_message += f"\n  Devices: {devices}"
        if names:
            full_message += f"\n  Objects: {names}"

        super().__init__(full_message)


class DtypeMismatchError(KinoPulseError):
    """Raised when tensors have incompatible dtypes.

    Args:
        message: Human-readable error description
        dtypes: List of dtypes encountered
        names: Names of objects for context

    Example:
        >>> raise DtypeMismatchError(
        ...     "Cannot operate on tensors with different dtypes",
        ...     dtypes=[torch.float32, torch.float64],
        ...     names=['state', 'params']
        ... )
    """

    def __init__(
        self,
        message: str,
        dtypes: Optional[list] = None,
        names: Optional[list] = None,
    ):
        self.dtypes = dtypes
        self.names = names

        full_message = message
        if dtypes:
            full_message += f"\n  Dtypes: {dtypes}"
        if names:
            full_message += f"\n  Objects: {names}"

        super().__init__(full_message)


class ConfigError(KinoPulseError):
    """Raised when configuration is invalid or malformed."""
    pass


class ModelInconsistencyError(KinoPulseError):
    """Raised when a system definition is malformed or inconsistent."""
    pass


class NotDifferentiableError(KinoPulseError):
    """Raised when an operation or system does not support gradients."""
    pass


class InvalidStateError(KinoPulseError):
    """Raised when a state violates constraints or is physically invalid."""
    pass


class ConvergenceError(KinoPulseError):
    """Raised when a solver or optimizer fails to converge."""
    pass


class ExportNotSupportedError(KinoPulseError):
    """Raised when a system or component cannot be exported."""
    pass


class HybridInconsistencyError(KinoPulseError):
    """Raised when hybrid system mode graph or transitions are inconsistent."""
    pass


class PDEGeometryError(KinoPulseError):
    """Raised when PDE geometry or discretization is invalid."""
    pass


def assert_same_device(*objs: Any, names: Optional[list[str]] = None) -> None:
    """Assert that all objects are on the same device.

    Args:
        *objs: Objects to check (tensors, modules, etc.)
        names: Optional names for error message context

    Raises:
        DeviceMismatchError: If objects are on different devices

    Example:
        >>> assert_same_device(state, input, names=['state', 'input'])
    """
    from .device import get_device  # Import here to avoid circular dependency

    devices = [get_device(obj) for obj in objs]
    devices = [d for d in devices if d is not None]

    if len(set(str(d) for d in devices)) > 1:
        raise DeviceMismatchError(
            "Objects on different devices",
            devices=devices,
            names=names,
        )


def assert_same_dtype(*objs: Any, names: Optional[list[str]] = None) -> None:
    """Assert that all objects have the same dtype.

    Args:
        *objs: Objects to check (tensors, etc.)
        names: Optional names for error message context

    Raises:
        DtypeMismatchError: If objects have different dtypes

    Example:
        >>> assert_same_dtype(state, input, names=['state', 'input'])
    """
    from .dtype import get_dtype  # Import here to avoid circular dependency

    dtypes = [get_dtype(obj) for obj in objs]
    dtypes = [d for d in dtypes if d is not None]

    if len(set(dtypes)) > 1:
        raise DtypeMismatchError(
            "Objects have different dtypes",
            dtypes=dtypes,
            names=names,
        )


def assert_finite(*tensors: Any, names: Optional[list[str]] = None) -> None:
    """Assert that all tensors contain only finite values (no NaN or Inf).

    Args:
        *tensors: Tensors to check
        names: Optional names for error message context

    Raises:
        InvalidStateError: If any tensor contains NaN or Inf

    Example:
        >>> assert_finite(state, input, names=['state', 'input'])
    """
    import torch

    for i, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            if not torch.isfinite(tensor).all():
                name = names[i] if names and i < len(names) else f"tensor_{i}"
                raise InvalidStateError(
                    f"Tensor '{name}' contains NaN or Inf values"
                )
