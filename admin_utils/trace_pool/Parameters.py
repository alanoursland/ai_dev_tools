"""
Parameter management for dynamical systems.

Provides Parameters class for managing system parameters with
support for trainable/fixed separation and hierarchical organization.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from kinopulse.utils import ConfigError
from kinopulse.utils.errors import InvalidStateError


@dataclass
class ParameterSpec:
    """Specification for a parameter.

    Attributes:
        name: Parameter name
        shape: Parameter shape
        dtype: Data type
        trainable: Whether parameter is trainable
        bounds: Optional (min, max) bounds
        positive: Whether parameter must be positive
        symmetric: Whether parameter matrix must be symmetric
        provenance: Origin of parameter ("user", "identified", "learned")
    """
    name: str
    shape: tuple
    dtype: torch.dtype
    trainable: bool = False
    bounds: Optional[tuple[float, float]] = None
    positive: bool = False
    symmetric: bool = False
    provenance: str = "user"

    def validate(self, param: torch.Tensor) -> bool:
        """Validate a parameter against this spec.

        Args:
            param: Parameter tensor to validate

        Returns:
            True if valid, False otherwise
        """
        # Check shape
        if param.shape != self.shape:
            return False

        # Check dtype
        if param.dtype != self.dtype:
            return False

        # Check bounds
        if self.bounds is not None:
            min_val, max_val = self.bounds
            if not (min_val <= param.min() and param.max() <= max_val):
                return False

        # Check positivity
        if self.positive and (param <= 0).any():
            return False

        # Check symmetry
        if self.symmetric and param.ndim == 2:
            if not torch.allclose(param, param.T):
                return False

        return True


class Parameters:
    """Dynamical system parameters.

    Supports flat dictionaries and hierarchical parameter sets with
    trainable/fixed separation.

    Example:
        >>> # Create from dict
        >>> params = Parameters({'mass': torch.tensor(1.0), 'k': torch.tensor(10.0)})
        >>> params['mass']
        tensor(1.0)
        >>>
        >>> # With specifications
        >>> specs = {
        ...     'mass': ParameterSpec('mass', (), torch.float32, trainable=True, positive=True),
        ...     'k': ParameterSpec('k', (), torch.float32, bounds=(0, 100))
        ... }
        >>> params = Parameters({'mass': torch.tensor(1.0), 'k': torch.tensor(10.0)}, specs)
    """

    def __init__(
        self,
        data: Optional[Union[Dict[str, torch.Tensor], nn.Module]] = None,
        specs: Optional[Dict[str, ParameterSpec]] = None,
        subsystem_params: Optional[Dict[str, 'Parameters']] = None,
        trainable: Optional[list[str]] = None,
    ):
        """Initialize parameters.

        Args:
            data: Parameter data (dict or nn.Module), defaults to empty dict
            specs: Optional parameter specifications
            subsystem_params: Optional subsystem parameters for hierarchical systems
            trainable: Optional list of trainable parameter names
        """
        self.specs = specs or {}
        self.subsystem_params = subsystem_params or {}
        self._trainable_names = set(trainable) if trainable else set()

        if data is None:
            self._data = {}
        elif isinstance(data, nn.Module):
            # Extract parameters from module
            self._data = {name: param for name, param in data.named_parameters()}
        elif isinstance(data, dict):
            self._data = data
        else:
            raise TypeError(f"data must be dict or nn.Module, got {type(data)}")

    def __getitem__(self, key: str) -> torch.Tensor:
        """Access parameter by name.

        Args:
            key: Parameter name

        Returns:
            Parameter tensor

        Raises:
            KeyError: If parameter not found
        """
        # Check hierarchical access (e.g., "subsystem.param")
        if '.' in key:
            subsystem, subkey = key.split('.', 1)
            if subsystem in self.subsystem_params:
                return self.subsystem_params[subsystem][subkey]

        if key not in self._data:
            raise KeyError(f"Parameter '{key}' not found")

        return self._data[key]

    def __len__(self) -> int:
        """Number of top-level parameters."""
        return len(self._data)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Set parameter value with validation.

        Args:
            key: Parameter name
            value: New parameter value

        Raises:
            ConfigError: If value violates constraints
        """
        # Validate against spec if available
        if key in self.specs:
            spec = self.specs[key]

            # Check shape
            if value.shape != spec.shape:
                raise ConfigError(
                    f"Parameter '{key}' shape mismatch: "
                    f"expected {spec.shape}, got {value.shape}"
                )

            # Check dtype
            if value.dtype != spec.dtype:
                raise ConfigError(
                    f"Parameter '{key}' dtype mismatch: "
                    f"expected {spec.dtype}, got {value.dtype}"
                )

            # Check bounds
            if spec.bounds is not None:
                min_val, max_val = spec.bounds
                if not (min_val <= value.min() and value.max() <= max_val):
                    raise ConfigError(
                        f"Parameter '{key}' violates bounds [{min_val}, {max_val}]"
                    )

            # Check positivity
            if spec.positive and (value <= 0).any():
                raise ConfigError(f"Parameter '{key}' must be positive")

            # Check symmetry
            if spec.symmetric and value.ndim == 2:
                if not torch.allclose(value, value.T):
                    raise ConfigError(f"Parameter '{key}' must be symmetric")

        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists.

        Args:
            key: Parameter name

        Returns:
            True if parameter exists
        """
        if '.' in key:
            subsystem, subkey = key.split('.', 1)
            if subsystem in self.subsystem_params:
                return subkey in self.subsystem_params[subsystem]
        return key in self._data

    def get(self, key: str, default=None):
        """Get parameter value with optional default.

        Args:
            key: Parameter name
            default: Default value if parameter not found

        Returns:
            Parameter value or default
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def data(self) -> Dict[str, torch.Tensor]:
        """Get parameter data dictionary.

        Returns:
            Dict of parameters
        """
        return self._data

    @property
    def trainable(self) -> Dict[str, torch.Tensor]:
        """Get only trainable parameters.

        Returns:
            Dict of trainable parameters
        """
        trainable = {}
        for key, value in self._data.items():
            # If trainable names were specified, use that
            if self._trainable_names:
                if key in self._trainable_names:
                    trainable[key] = value
            # Otherwise check requires_grad or spec
            else:
                # Check if has requires_grad
                if hasattr(value, 'requires_grad') and value.requires_grad:
                    trainable[key] = value
                # Check spec
                elif key in self.specs and self.specs[key].trainable:
                    trainable[key] = value
                # Default: if no spec and no trainable list, all are trainable
                elif not self.specs and not self._trainable_names:
                    trainable[key] = value
        return trainable

    @property
    def fixed(self) -> Dict[str, torch.Tensor]:
        """Get only fixed parameters.

        Returns:
            Dict of fixed parameters
        """
        fixed = {}
        trainable_keys = set(self.trainable.keys())
        for key, value in self._data.items():
            if key not in trainable_keys:
                fixed[key] = value
        return fixed

    @property
    def all(self) -> Dict[str, torch.Tensor]:
        """Get all parameters.

        Returns:
            Dict of all parameters
        """
        return self._data.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Export as flat dictionary.

        Returns:
            Dictionary of parameter names to values
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy().tolist()
            else:
                result[key] = value
        return result

    def validate(self) -> bool:
        """Check bounds, constraints, positivity.

        Returns:
            True if all constraints satisfied

        Raises:
            InvalidStateError: If any constraint is violated
        """
        for key in self._data.keys():
            try:
                # Try to set the value (triggers validation)
                self[key] = self._data[key]
            except ConfigError as e:
                raise InvalidStateError(str(e)) from e
        return True

    def flatten(self) -> torch.Tensor:
        """Flatten to single vector for optimization.

        Returns:
            Flattened parameter vector
        """
        tensors = []
        for key in sorted(self._data.keys()):
            param = self._data[key]
            tensors.append(param.flatten())
        return torch.cat(tensors)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameters':
        """Create parameters from dictionary.

        Args:
            data: Dictionary with parameter values

        Returns:
            Parameters instance
        """
        tensor_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensor_data[key] = value
            else:
                tensor_data[key] = torch.tensor(value)
        return cls(tensor_data)

    def keys(self):
        """Get parameter names."""
        return self._data.keys()

    def values(self):
        """Get parameter values."""
        return self._data.values()

    def items(self):
        """Get parameter (name, value) pairs."""
        return self._data.items()

    def update(self, other: Union[Dict[str, torch.Tensor], 'Parameters']) -> None:
        """Update parameters from dict or another Parameters object.

        Args:
            other: Dict or Parameters to update from
        """
        if isinstance(other, Parameters):
            self._data.update(other._data)
        elif isinstance(other, dict):
            self._data.update(other)
        else:
            raise TypeError(f"Cannot update from {type(other)}")

    def clone(self) -> 'Parameters':
        """Create a deep copy of parameters.

        Returns:
            Cloned Parameters instance
        """
        cloned_data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                       for k, v in self._data.items()}
        cloned = Parameters(cloned_data, self.specs, self.subsystem_params)
        cloned._trainable_names = self._trainable_names.copy()
        return cloned

    def to(self, device: torch.device) -> 'Parameters':
        """Move parameters to device.

        Args:
            device: Target device

        Returns:
            New Parameters instance on target device
        """
        moved_data = {}
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                moved_data[key] = value.to(device)
            else:
                moved_data[key] = value

        result = Parameters(moved_data, self.specs, self.subsystem_params)
        result._trainable_names = self._trainable_names.copy()
        return result

    @property
    def device(self) -> Optional[torch.device]:
        """Get device of parameters.

        Returns:
            Device if all parameters on same device, None if empty or mixed
        """
        devices = set()
        for value in self._data.values():
            if isinstance(value, torch.Tensor):
                devices.add(value.device)

        if len(devices) == 0:
            return None
        elif len(devices) == 1:
            return devices.pop()
        else:
            # Mixed devices, return first one
            return next(iter(devices))
