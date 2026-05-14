"""
Solver configuration structures.

Provides configuration dataclasses for all solver options including tolerances,
step sizes, events, adjoints, and diagnostics.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Literal
import torch


class SolverMethod(Enum):
    """Enumeration of available solver methods."""

    # Explicit methods
    EULER = "euler"
    MIDPOINT = "midpoint"
    RK2 = "rk2"
    RK4 = "rk4"
    RK45 = "rk45"
    DOPRI5 = "dopri5"
    TSIT5 = "tsit5"

    # Implicit methods
    IMPLICIT_EULER = "implicit_euler"
    IMPLICIT_MIDPOINT = "implicit_midpoint"
    BDF = "bdf"
    RADAU = "radau"

    # Automatic selection
    AUTO = "auto"


@dataclass
class ToleranceConfig:
    """Tolerance configuration for adaptive solvers.

    Attributes:
        atol: Absolute tolerance (default: 1e-6)
        rtol: Relative tolerance (default: 1e-3)
        norm: Norm type for error computation ('L2', 'Linf', 'weighted')
    """

    atol: float = 1e-6
    rtol: float = 1e-3
    norm: Literal["L2", "Linf", "weighted"] = "L2"

    def __post_init__(self):
        """Validate tolerance parameters."""
        if self.atol < 0:
            raise ValueError(f"atol must be non-negative, got {self.atol}")
        if self.rtol < 0:
            raise ValueError(f"rtol must be non-negative, got {self.rtol}")
        if self.norm not in ["L2", "Linf", "weighted"]:
            raise ValueError(f"norm must be 'L2', 'Linf', or 'weighted', got {self.norm}")


@dataclass
class StepSizeConfig:
    """Step size configuration for time integration.

    Attributes:
        dt: Initial/fixed time step size (default: 0.01)
        dt_min: Minimum allowed step size (default: 1e-10)
        dt_max: Maximum allowed step size (default: 1.0)
        adaptive: Enable adaptive stepping (default: False)
        safety_factor: Safety factor for step size adaptation (default: 0.9)
        controller: Step size controller type ('PI', 'I', 'predictive')
    """

    dt: float = 0.01
    dt_min: float = 1e-10
    dt_max: float = 1.0
    adaptive: bool = False
    safety_factor: float = 0.9
    controller: Literal["PI", "I", "predictive"] = "I"

    def __post_init__(self):
        """Validate step size parameters."""
        if self.dt_min >= self.dt_max:
            raise ValueError(
                f"dt_min must be less than dt_max, got {self.dt_min} >= {self.dt_max}"
            )
        # Allow negative dt for backward integration by checking magnitude only
        magnitude = abs(self.dt)
        if magnitude < self.dt_min or magnitude > self.dt_max:
            raise ValueError(
                f"|dt| must be between dt_min and dt_max, got {self.dt} not in "
                f"[{self.dt_min}, {self.dt_max}]"
            )
        if self.safety_factor <= 0 or self.safety_factor > 1:
            raise ValueError(
                f"safety_factor must be in (0, 1], got {self.safety_factor}"
            )


@dataclass
class EventConfig:
    """Event detection configuration.

    Attributes:
        enabled: Enable event detection (default: False)
        tolerance: Tolerance for event location (default: 1e-8)
        max_refinements: Maximum bisection refinements (default: 10)
        direction: Event direction ('rising', 'falling', 'both')
    """

    enabled: bool = False
    tolerance: float = 1e-8
    max_refinements: int = 10
    direction: Literal["rising", "falling", "both"] = "both"

    def __post_init__(self):
        """Validate event parameters."""
        if self.direction not in ["rising", "falling", "both"]:
            raise ValueError(
                f"direction must be 'rising', 'falling', or 'both', got {self.direction}"
            )
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.max_refinements < 1:
            raise ValueError(
                f"max_refinements must be at least 1, got {self.max_refinements}"
            )


@dataclass
class AdjointConfig:
    """Adjoint sensitivity configuration.

    Attributes:
        enabled: Enable adjoint sensitivity computation (default: False)
        method: Adjoint method type ('continuous', 'discrete', 'direct')
        checkpoint_steps: Number of checkpoints for memory optimization (default: 100)
        rtol: Relative tolerance for adjoint solve (default: 1e-6)
        atol: Absolute tolerance for adjoint solve (default: 1e-8)
    """

    enabled: bool = False
    method: Literal["continuous", "discrete", "direct"] = "continuous"
    checkpoint_steps: int = 100
    rtol: float = 1e-6
    atol: float = 1e-8

    def __post_init__(self):
        """Validate adjoint parameters."""
        if self.method not in ["continuous", "discrete", "direct"]:
            raise ValueError(
                f"method must be 'continuous', 'discrete', or 'direct', got {self.method}"
            )
        if self.checkpoint_steps < 1:
            raise ValueError(
                f"checkpoint_steps must be at least 1, got {self.checkpoint_steps}"
            )


@dataclass
class DiagnosticConfig:
    """Diagnostic tracking configuration.

    Attributes:
        enabled: Enable diagnostic tracking (default: True)
        log_steps: Log every Nth step (default: 1)
        log_errors: Track error estimates (default: True)
        log_events: Track event occurrences (default: True)
        track_invariants: Track system invariants (default: False)
        warn_on_failure: Issue warnings on step failures (default: True)
    """

    enabled: bool = True
    log_steps: int = 1
    log_errors: bool = True
    log_events: bool = True
    track_invariants: bool = False
    warn_on_failure: bool = True

    def __post_init__(self):
        """Validate diagnostic parameters."""
        if self.log_steps < 1:
            raise ValueError(f"log_steps must be at least 1, got {self.log_steps}")


@dataclass
class SolverConfig:
    """Master solver configuration.

    Combines all configuration options for solver execution.

    Attributes:
        method: Solver method (default: SolverMethod.AUTO)
        tolerance: Tolerance configuration
        step_size: Step size configuration
        events: Event detection configuration
        adjoint: Adjoint sensitivity configuration
        diagnostics: Diagnostic tracking configuration
        device: Torch device for computation (default: CPU)
        dtype: Torch data type (default: float32)
        save_every: Save every Nth step to trajectory (default: 1)
        dense_output: Enable dense output interpolation (default: False)
        max_steps: Maximum number of integration steps (default: 1000000)
        batch_mode: Enable batch processing (default: False)

    Example:
        >>> config = SolverConfig(
        ...     method=SolverMethod.RK4,
        ...     step_size=StepSizeConfig(dt=0.01),
        ...     tolerance=ToleranceConfig(atol=1e-8, rtol=1e-6)
        ... )
    """

    method: SolverMethod = SolverMethod.AUTO
    tolerance: Optional[ToleranceConfig] = None
    step_size: Optional[StepSizeConfig] = None
    events: Optional[EventConfig] = None
    adjoint: Optional[AdjointConfig] = None
    diagnostics: Optional[DiagnosticConfig] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    save_every: int = 1
    dense_output: bool = False
    max_steps: int = 1000000
    batch_mode: bool = False

    def __init__(
        self,
        method: Optional[SolverMethod] = SolverMethod.AUTO,
        tolerance: Optional[ToleranceConfig] = None,
        step_size: Optional[StepSizeConfig] = None,
        events: Optional[EventConfig] = None,
        adjoint: Optional[AdjointConfig] = None,
        diagnostics: Optional[DiagnosticConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        save_every: int = 1,
        dense_output: bool = False,
        max_steps: int = 1000000,
        batch_mode: bool = False,
        **kwargs,
    ):
        """
        Custom initializer that accepts common shorthand kwargs (dt, rtol, atol, etc.).

        This keeps backward compatibility with convenience functions and tests
        that pass scalar values directly instead of nested config objects.
        """
        step_overrides = {
            key: kwargs.pop(key)
            for key in ["dt", "adaptive", "dt_min", "dt_max", "safety_factor", "controller"]
            if key in kwargs
        }
        tol_overrides = {
            key: kwargs.pop(key)
            for key in ["rtol", "atol", "norm"]
            if key in kwargs
        }

        self.method = self._coerce_method(method)
        # Handle dtype: use float32 if not specified
        self.dtype = dtype
        self.step_size = self._coerce_step_size(step_size, step_overrides)
        self.tolerance = self._coerce_tolerance(tolerance, tol_overrides)
        self.events = events
        self.adjoint = adjoint
        self.diagnostics = diagnostics
        self.device = device
        self.save_every = save_every
        self.dense_output = dense_output
        self.max_steps = max_steps
        self.batch_mode = batch_mode

        # Preserve any extra kwargs as attributes for forward compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self):
        """Initialize nested configs with defaults if not provided."""
        # Normalize method (accept strings)
        self.method = self._coerce_method(self.method)

        # Convert dicts or missing configs into concrete config objects
        if self.tolerance is None:
            self.tolerance = ToleranceConfig()
        elif isinstance(self.tolerance, dict):
            self.tolerance = ToleranceConfig(**self.tolerance)

        if self.step_size is None:
            self.step_size = StepSizeConfig()
        elif isinstance(self.step_size, dict):
            self.step_size = StepSizeConfig(**self.step_size)

        if self.events is None:
            self.events = EventConfig()
        elif isinstance(self.events, bool):
            self.events = EventConfig(enabled=self.events)
        elif isinstance(self.events, dict):
            self.events = EventConfig(**self.events)

        if self.adjoint is None:
            self.adjoint = AdjointConfig()
        elif isinstance(self.adjoint, dict):
            self.adjoint = AdjointConfig(**self.adjoint)

        if self.diagnostics is None:
            self.diagnostics = DiagnosticConfig()
        elif isinstance(self.diagnostics, dict):
            self.diagnostics = DiagnosticConfig(**self.diagnostics)

        if self.device is None:
            self.device = torch.device("cpu")
        elif not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        # Validate max_steps
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {self.max_steps}")

        # Validate save_every
        if self.save_every < 1:
            raise ValueError(f"save_every must be at least 1, got {self.save_every}")

    # ------------------------------------------------------------------
    # Convenience property accessors for common nested config fields
    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        """Alias to step_size.dt for compatibility."""
        return self.step_size.dt

    @dt.setter
    def dt(self, value: float):
        self.step_size.dt = value

    @property
    def dt_min(self) -> float:
        return self.step_size.dt_min

    @property
    def dt_max(self) -> float:
        return self.step_size.dt_max

    @property
    def adaptive(self) -> bool:
        return self.step_size.adaptive

    @adaptive.setter
    def adaptive(self, value: bool):
        self.step_size.adaptive = value

    @property
    def rtol(self) -> float:
        return self.tolerance.rtol

    @property
    def atol(self) -> float:
        return self.tolerance.atol

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_method(method: Optional[SolverMethod]) -> SolverMethod:
        """Accept SolverMethod or string value (case-insensitive)."""
        if isinstance(method, SolverMethod):
            return method
        if isinstance(method, str):
            try:
                return SolverMethod(method.lower())
            except ValueError as e:
                valid = ", ".join(m.value for m in SolverMethod)
                raise ValueError(f"Unknown solver method '{method}'. Valid methods: {valid}") from e
        return SolverMethod.AUTO

    @staticmethod
    def _coerce_step_size(step_size, overrides) -> Optional[StepSizeConfig]:
        """Normalize step_size input into a StepSizeConfig."""
        if isinstance(step_size, StepSizeConfig):
            return replace(step_size, **overrides) if overrides else step_size
        if isinstance(step_size, dict):
            merged = {**step_size, **overrides}
            return StepSizeConfig(**merged)
        if isinstance(step_size, (float, int)):
            overrides.setdefault("dt", float(step_size))
            return StepSizeConfig(**overrides)
        if step_size is None:
            return StepSizeConfig(**overrides) if overrides else None
        raise TypeError(f"step_size must be StepSizeConfig, dict, numeric, or None, got {type(step_size)}")

    @staticmethod
    def _coerce_tolerance(tolerance, overrides) -> Optional[ToleranceConfig]:
        """Normalize tolerance input into a ToleranceConfig."""
        if isinstance(tolerance, ToleranceConfig):
            return replace(tolerance, **overrides) if overrides else tolerance
        if isinstance(tolerance, dict):
            merged = {**tolerance, **overrides}
            return ToleranceConfig(**merged)
        if tolerance is None:
            return ToleranceConfig(**overrides) if overrides else None
        raise TypeError(f"tolerance must be ToleranceConfig, dict, or None, got {type(tolerance)}")
