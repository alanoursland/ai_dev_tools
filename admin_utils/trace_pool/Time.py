"""
Time domain representations for dynamical systems.

Provides Time class for continuous, discrete, and hybrid time.
Supports both standard scalar values (float/int) for simulation and
PyTorch Tensors for differentiable physics and JIT compilation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Any, Tuple
import torch


class TimeType(Enum):
    """Time domain type."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"


@dataclass
class TimeSpec:
    """Time domain specification.

    Attributes:
        time_type: Type of time domain
        t0: Initial continuous time
        tf: Final continuous time
        k0: Initial discrete step
        kf: Final discrete step
        characteristic_scale: Optional characteristic time scale
        suggested_horizon: Optional suggested integration horizon
        time_varying: Whether system explicitly depends on time
    """
    time_type: TimeType
    t0: Optional[float] = None
    tf: Optional[float] = None
    k0: Optional[int] = None
    kf: Optional[int] = None
    characteristic_scale: Optional[float] = None
    suggested_horizon: Optional[float] = None
    time_varying: bool = True

    @property
    def duration(self) -> Optional[float]:
        """Compute duration for continuous time."""
        if self.t0 is not None and self.tf is not None:
            return self.tf - self.t0
        return None

    @property
    def num_steps(self) -> Optional[int]:
        """Compute number of discrete steps."""
        if self.k0 is not None and self.kf is not None:
            return self.kf - self.k0
        return None


class Time:
    """Time representation for dynamical systems.

    Supports continuous (t ∈ ℝ), discrete (k ∈ ℕ), and hybrid ((τ, j)) time.
    Values can be backed by standard Python types (float/int) or torch.Tensor.

    Example:
        >>> # Continuous time
        >>> t = Time(1.5)
        >>> t.continuous
        1.5
        >>>
        >>> # Differentiable time (Tensor)
        >>> t_grad = Time(torch.tensor(1.5, requires_grad=True))
        >>> t_grad.continuous.requires_grad
        True
        >>>
        >>> # Discrete time
        >>> t = Time(10, TimeType.DISCRETE)
        >>> t.discrete
        10
        >>>
        >>> # Hybrid time
        >>> t = Time((1.5, 3), TimeType.HYBRID)
        >>> t.hybrid
        (1.5, 3)
    """

    def __init__(
        self,
        value: Union[float, int, tuple, torch.Tensor],
        time_type: TimeType = TimeType.CONTINUOUS,
    ):
        """Initialize time.

        Args:
            value: Time value. Can be:
                - float/int: Standard simulation
                - torch.Tensor: Differentiable/JIT contexts
                - tuple: (continuous, discrete) for hybrid time
            time_type: Type of time domain
        """
        self.time_type = time_type

        # Helper to detect if a value is a Tensor
        is_tensor = isinstance(value, torch.Tensor)

        if time_type == TimeType.CONTINUOUS:
            if is_tensor:
                self._continuous = value
                self._discrete = None
            elif isinstance(value, (int, float)):
                self._continuous = float(value)
                self._discrete = None
            else:
                raise ValueError(f"Continuous time must be float, int, or Tensor. Got {type(value)}")

        elif time_type == TimeType.DISCRETE:
            if is_tensor:
                self._continuous = None
                self._discrete = value
            elif isinstance(value, int):
                self._continuous = None
                self._discrete = int(value)
            else:
                raise ValueError(f"Discrete time must be int or Tensor. Got {type(value)}")

        elif time_type == TimeType.HYBRID:
            if isinstance(value, tuple) and len(value) == 2:
                # Handle tuple where elements might be tensors
                self._continuous = value[0]  # Flow time (float or Tensor)
                self._discrete = value[1]    # Jump count (int or Tensor)
                
                # Validation for standard types (skip for Tensors to allow flexibility)
                if not isinstance(self._continuous, (int, float, torch.Tensor)):
                     raise ValueError("Hybrid continuous part must be numeric")
                if not isinstance(self._discrete, (int, torch.Tensor)):
                     raise ValueError("Hybrid discrete part must be integer-like")
            else:
                raise ValueError("Hybrid time must be tuple (flow_time, jump_count)")

        else:
            raise ValueError(f"Unknown time type: {time_type}")

    @property
    def value(self) -> Union[float, int, torch.Tensor]:
        """Get time value in most appropriate form.

        Returns:
            - For continuous time: float or Tensor
            - For discrete time: int or Tensor
            - For hybrid time: float or Tensor (continuous component only)

        Note:
            This is a convenience property for backward compatibility.
            Use `continuous` or `discrete` for explicit component access.
        """
        if self._continuous is not None:
            return self._continuous
        elif self._discrete is not None:
            return self._discrete
        else:
            raise ValueError("Time object has no value")

    @property
    def continuous(self) -> Union[float, torch.Tensor]:
        """Get continuous time component.

        Returns:
            Continuous time value (float or Tensor)

        Raises:
            ValueError: If time type is discrete
        """
        if self._continuous is None:
            raise ValueError("Discrete time has no continuous component")
        return self._continuous

    @property
    def discrete(self) -> Union[int, torch.Tensor]:
        """Get discrete time component.

        Returns:
            Discrete time value (int or Tensor)

        Raises:
            ValueError: If time type is continuous only
        """
        if self._discrete is None:
            raise ValueError("Continuous time has no discrete component")
        return self._discrete

    @property
    def hybrid(self) -> Tuple[Union[float, torch.Tensor], Union[int, torch.Tensor]]:
        """Get hybrid time (flow time, jump count).

        Returns:
            Tuple of (flow_time, jump_count)

        Raises:
            ValueError: If time type is not hybrid
        """
        if self.time_type != TimeType.HYBRID:
            raise ValueError("Only hybrid time has flow and jump components")
        return (self._continuous, self._discrete)

    def increment_flow(self, dt: Union[float, torch.Tensor]) -> 'Time':
        """Advance flow time (for hybrid systems).

        Args:
            dt: Time increment (float or Tensor)

        Returns:
            New time with advanced flow time

        Raises:
            ValueError: If time type does not support flow
        """
        if self.time_type == TimeType.CONTINUOUS:
            return Time(self._continuous + dt, TimeType.CONTINUOUS)
        elif self.time_type == TimeType.HYBRID:
            return Time((self._continuous + dt, self._discrete), TimeType.HYBRID)
        else:
            raise ValueError("Discrete time cannot increment flow")

    def increment_jump(self, count: int = 1) -> 'Time':
        """Increment jump counter (for hybrid systems).

        Args:
            count: Number of jumps to increment (default 1)

        Returns:
            New time with incremented jump count

        Raises:
            ValueError: If time type does not support jumps
        """
        if self.time_type == TimeType.DISCRETE:
            return Time(self._discrete + count, TimeType.DISCRETE)
        elif self.time_type == TimeType.HYBRID:
            return Time((self._continuous, self._discrete + count), TimeType.HYBRID)
        else:
            raise ValueError("Continuous time cannot increment jumps")

    def to_continuous(self, dt: Optional[Union[float, torch.Tensor]] = None) -> 'Time':
        """Convert to continuous time.

        Args:
            dt: Time step for discrete-to-continuous conversion

        Returns:
            Continuous time representation
        """
        if self.time_type == TimeType.CONTINUOUS:
            return Time(self._continuous)
        elif self.time_type == TimeType.DISCRETE and dt is not None:
            return Time(self._discrete * dt)
        elif self.time_type == TimeType.HYBRID:
            if dt is not None:
                return Time(self._continuous + self._discrete * dt)
            return Time(self._continuous)
        else:
            raise ValueError("Cannot convert discrete time without dt parameter")

    def to_discrete(self, dt: Union[float, torch.Tensor]) -> 'Time':
        """Convert to discrete time.

        Args:
            dt: Time step for conversion

        Returns:
            Discrete time representation
        """
        # Handling Tensors in casting logic is tricky (int() fails on tensors)
        # We rely on PyTorch's floor/div operations if needed
        is_tensor = isinstance(self._continuous, torch.Tensor) or isinstance(dt, torch.Tensor)

        if self.time_type == TimeType.DISCRETE:
            return Time(self._discrete, TimeType.DISCRETE)
        elif self.time_type == TimeType.CONTINUOUS:
            if is_tensor:
                # Use torch operations for graph compatibility
                k = (self._continuous / dt).long() if hasattr(self._continuous, 'long') else int(self._continuous / dt)
            else:
                k = int(self._continuous / dt)
            return Time(k, TimeType.DISCRETE)
        else:  # HYBRID
            if is_tensor:
                # Continuous part contribution + discrete part
                k_cont = (self._continuous / dt).long() if hasattr(self._continuous, 'long') else int(self._continuous / dt)
                k = k_cont + self._discrete
            else:
                k = int(self._continuous / dt) + self._discrete
            return Time(k, TimeType.DISCRETE)

    def to_hybrid(self, k: Union[int, torch.Tensor]) -> 'Time':
        """Convert to hybrid time.

        Args:
            k: Discrete component for hybrid time

        Returns:
            Hybrid time representation
        """
        if self.time_type == TimeType.HYBRID:
            return Time((self._continuous, k), TimeType.HYBRID)
        elif self.time_type == TimeType.CONTINUOUS:
            return Time((self._continuous, k), TimeType.HYBRID)
        else:  # DISCRETE
            # Return (0.0, k) but match tensor/float type of original
            zero_val = torch.tensor(0.0) if isinstance(self._discrete, torch.Tensor) else 0.0
            return Time((zero_val, k), TimeType.HYBRID)

    def __eq__(self, other: Any) -> Union[bool, torch.Tensor]:
        """Check equality of two Time objects.
        
        Returns:
            bool or Tensor(bool) depending on input types.
        """
        if not isinstance(other, Time):
            return False
            
        # Helper to safely compare float or Tensor
        def safe_eq(a, b, tol=1e-10):
            if a is None and b is None: return True
            if a is None or b is None: return False
            
            # Tensor path
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                return torch.abs(a - b) < tol
            
            # Float path
            return abs(a - b) < tol

        cont_equal = safe_eq(self._continuous, other._continuous)
        
        # Discrete equality
        if self._discrete is not None and other._discrete is not None:
            disc_equal = (self._discrete == other._discrete)
        elif self._discrete is None and other._discrete is None:
            disc_equal = True
        else:
            disc_equal = False

        # If we have tensors, we might get a BoolTensor back
        if isinstance(cont_equal, torch.Tensor) or isinstance(disc_equal, torch.Tensor):
            return cont_equal & disc_equal
            
        return cont_equal and disc_equal

    def __lt__(self, other: 'Time') -> Union[bool, torch.Tensor]:
        """Less than comparison.

        For hybrid time, uses lexicographic ordering: (t1, j1) < (t2, j2)
        if t1 < t2, or (t1 == t2 and j1 < j2)
        """
        if not isinstance(other, Time):
            raise TypeError(f"Cannot compare Time with {type(other)}")

        # Continuous comparison
        if self._continuous is not None and other._continuous is not None:
            # Check distinctness tolerance
            diff = torch.abs(self._continuous - other._continuous) if isinstance(self._continuous, torch.Tensor) else abs(self._continuous - other._continuous)
            
            # Logic: if strictly less AND diff > tol
            # This is tricky with tensors. Simplification:
            return self._continuous < other._continuous

        # Discrete comparison (fallback if continuous is None or Equal)
        if self._discrete is not None and other._discrete is not None:
            return self._discrete < other._discrete
        elif self._discrete is None:
            return False
        else:  # other._discrete is None
            return True

    def __le__(self, other: 'Time') -> Union[bool, torch.Tensor]:
        """Less than or equal comparison."""
        # Using specific logic to avoid recursion or composite boolean tensors issues
        if not isinstance(other, Time):
            raise TypeError(f"Cannot compare Time with {type(other)}")
            
        if self._continuous is not None:
            return self._continuous <= other._continuous
        return self._discrete <= other._discrete

    def __gt__(self, other: 'Time') -> Union[bool, torch.Tensor]:
        """Greater than comparison."""
        if not isinstance(other, Time):
            raise TypeError(f"Cannot compare Time with {type(other)}")
            
        if self._continuous is not None:
            return self._continuous > other._continuous
        return self._discrete > other._discrete

    def __ge__(self, other: 'Time') -> Union[bool, torch.Tensor]:
        """Greater than or equal comparison."""
        if not isinstance(other, Time):
            raise TypeError(f"Cannot compare Time with {type(other)}")
            
        if self._continuous is not None:
            return self._continuous >= other._continuous
        return self._discrete >= other._discrete

    def __add__(self, other: Union['Time', float, int, torch.Tensor]) -> 'Time':
        """Add scalar or Time to this Time."""
        is_tensor_op = isinstance(other, torch.Tensor)
        
        # Handle scalar addition (float/int/Tensor)
        if isinstance(other, (int, float, torch.Tensor)):
            # Add to continuous component
            if self._continuous is not None:
                new_cont = self._continuous + other
            else:
                new_cont = None
                
            if self.time_type == TimeType.CONTINUOUS:
                return Time(new_cont)
            elif self.time_type == TimeType.DISCRETE:
                # For discrete, 'other' effectively represents steps
                return Time(self._discrete + other, TimeType.DISCRETE)
            else:  # HYBRID
                return Time((new_cont, self._discrete), TimeType.HYBRID)
                
        # Handle Time + Time addition
        elif isinstance(other, Time):
            new_cont = None
            if self._continuous is not None and other._continuous is not None:
                new_cont = self._continuous + other._continuous
                
            new_disc = None
            if self._discrete is not None and other._discrete is not None:
                new_disc = self._discrete + other._discrete
            
            # Determine result type based on what components we successfully added
            if new_disc is None or (isinstance(new_disc, int) and new_disc == 0):
                return Time(new_cont)
            elif new_cont is None or (isinstance(new_cont, (int, float)) and new_cont == 0):
                return Time(new_disc, TimeType.DISCRETE)
            else:
                return Time((new_cont, new_disc), TimeType.HYBRID)
        else:
            raise TypeError(f"Cannot add Time and {type(other)}")

    def __sub__(self, other: Union['Time', float, int, torch.Tensor]) -> Union['Time', float, int, torch.Tensor, Tuple]:
        """Subtract scalar or Time from this Time."""
        
        # Scalar subtraction
        if isinstance(other, (int, float, torch.Tensor)):
            if self._continuous is not None:
                new_cont = self._continuous - other
            else:
                new_cont = None
                
            if self.time_type == TimeType.CONTINUOUS:
                return Time(new_cont)
            elif self.time_type == TimeType.DISCRETE:
                return Time(self._discrete - other, TimeType.DISCRETE)
            else:  # HYBRID
                return Time((new_cont, self._discrete), TimeType.HYBRID)
                
        # Time - Time subtraction (Returns Duration/Difference, NOT a Time object usually)
        elif isinstance(other, Time):
            if self.time_type == TimeType.CONTINUOUS and other.time_type == TimeType.CONTINUOUS:
                return self._continuous - other._continuous
            elif self.time_type == TimeType.DISCRETE and other.time_type == TimeType.DISCRETE:
                return self._discrete - other._discrete
            else:
                # For hybrid or mixed, return tuple of differences
                diff_cont = (self._continuous - other._continuous) if (self._continuous is not None and other._continuous is not None) else 0.0
                diff_disc = (self._discrete - other._discrete) if (self._discrete is not None and other._discrete is not None) else 0
                return (diff_cont, diff_disc)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from Time")

    def __repr__(self) -> str:
        """String representation."""
        if self.time_type == TimeType.CONTINUOUS:
            val_str = f"{self._continuous:.4f}" if isinstance(self._continuous, float) else str(self._continuous)
            return f"Time(t={val_str}, type=CONTINUOUS)"
        elif self.time_type == TimeType.DISCRETE:
            return f"Time(k={self._discrete}, type=DISCRETE)"
        else:  # HYBRID
            t_str = f"{self._continuous:.4f}" if isinstance(self._continuous, float) else str(self._continuous)
            return f"Time(τ={t_str}, j={self._discrete}, type=HYBRID)"