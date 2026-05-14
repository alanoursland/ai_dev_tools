"""
Dynamical system protocol and base abstractions.

Defines the DynamicalSystem protocol that all system types must satisfy.
"""

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
import torch

from .state import State, StateSpec
from .time import Time
from .parameters import Parameters, ParameterSpec
from .capabilities import Capabilities
from .metadata import SystemMetadata


@runtime_checkable
class DynamicalSystem(Protocol):
    """Protocol for all dynamical systems in KinoPulse.

    All system types (symbolic, numeric, neural, hybrid, PDE, composite)
    must satisfy this protocol.
    """

    capabilities: Capabilities
    metadata: SystemMetadata

    def dynamics(
        self,
        t: Time,
        x: State,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> torch.Tensor:
        """Evaluate system dynamics: dx/dt = f(t, x, u, params).

        Args:
            t: Current time
            x: Current state
            u: Optional input/control
            params: Optional parameters

        Returns:
            State derivative dx/dt
        """
        ...

    def output(
        self,
        t: Time,
        x: State,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> torch.Tensor:
        """Evaluate system output: y = h(t, x, u, params).

        Args:
            t: Current time
            x: Current state
            u: Optional input/control
            params: Optional parameters

        Returns:
            Output y
        """
        ...

    # Optional methods with defaults

    def jacobian(
        self,
        t: Time,
        x: State,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute Jacobians: (∂f/∂x, ∂f/∂u).

        Only implemented if capabilities.jacobian != "none".
        """
        raise NotImplementedError()

    def residual(
        self,
        t: Time,
        x: State,
        dx: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> torch.Tensor:
        """Evaluate DAE residual: F(t, x, dx, u, params) = 0.

        Only for DAE systems.
        """
        raise NotImplementedError()

    def events(
        self,
        t: Time,
        x: State,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> torch.Tensor:
        """Evaluate event functions for zero-crossing detection.

        Only if capabilities.event_support == True.
        """
        raise NotImplementedError()

    def reset(
        self,
        x_minus: State,
        mode_from: Optional[int] = None,
        mode_to: Optional[int] = None,
        params: Optional[Parameters] = None,
    ) -> State:
        """Apply reset map at discrete transition: x⁺ = R(x⁻, mode_from, mode_to).

        Only for hybrid systems.
        """
        raise NotImplementedError()

    def describe(self) -> str:
        """Get human-readable system description."""
        return f"{self.__class__.__name__}()"


def create_system(
    dynamics_fn: Any,
    output_fn: Optional[Any] = None,
    **kwargs: Any,
) -> DynamicalSystem:
    """Factory function for creating systems from functions.

    Args:
        dynamics_fn: Dynamics function
        output_fn: Optional output function
        **kwargs: Additional system properties

    Returns:
        System instance

    Note:
        Full implementation requires concrete system classes.
        For now, use direct instantiation of specific system types.
    """
    # Placeholder - full implementation requires concrete system classes
    raise NotImplementedError(
        "System factory requires concrete system implementations. "
        "Use specific system classes (e.g., NumericODE, SymbolicODE) directly."
    )


def create_state(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    spec: Optional[StateSpec] = None,
) -> State:
    """Factory function for creating states.

    Args:
        data: State data (tensor or dict)
        spec: Optional state specification

    Returns:
        State instance

    Example:
        >>> # Create from tensor
        >>> x = create_state(torch.randn(4, 10))
        >>>
        >>> # Create with spec
        >>> spec = StateSpec(
        ...     shape=(4, 10),
        ...     dtype=torch.float32,
        ...     device=torch.device('cpu')
        ... )
        >>> x = create_state(torch.randn(4, 10), spec)
    """
    state = State(data, spec=spec)

    # Validate if spec provided
    if spec is not None:
        state.validate()

    return state


def zero_state(
    spec: StateSpec,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> State:
    """Create zero-initialized state.

    Args:
        spec: State specification
        batch_size: Batch size (default 1)
        device: Optional device override
        dtype: Optional dtype override

    Returns:
        Zero-initialized state

    Example:
        >>> spec = StateSpec(
        ...     shape=(1, 10),
        ...     dtype=torch.float32,
        ...     device=torch.device('cpu')
        ... )
        >>> x = zero_state(spec, batch_size=4)
        >>> x.shape
        torch.Size([4, 10])
    """
    # Use spec device/dtype or overrides
    target_device = device if device is not None else spec.device
    target_dtype = dtype if dtype is not None else spec.dtype

    # Create shape with batch dimension
    if len(spec.shape) > 0 and spec.shape[0] != batch_size:
        # Replace first dimension with batch_size
        shape = (batch_size,) + spec.shape[1:]
    else:
        shape = (batch_size,) + spec.shape

    # Create zero tensor
    data = torch.zeros(shape, device=target_device, dtype=target_dtype)

    return State(data, spec=spec)


def create_parameters(
    data: Dict[str, torch.Tensor],
    trainable: Optional[List[str]] = None,
    specs: Optional[Dict[str, ParameterSpec]] = None,
) -> Parameters:
    """Factory function for creating parameters.

    Args:
        data: Dict of parameter tensors
        trainable: Optional list of trainable parameter names
        specs: Optional parameter specifications

    Returns:
        Parameters instance

    Example:
        >>> params = create_parameters({
        ...     'mass': torch.tensor(1.0),
        ...     'k': torch.tensor(10.0)
        ... }, trainable=['mass'])
    """
    # Mark trainable parameters
    if trainable:
        for name in trainable:
            if name in data and isinstance(data[name], torch.Tensor):
                data[name].requires_grad_(True)

    return Parameters(data, specs=specs)
