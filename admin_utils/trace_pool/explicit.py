"""
Explicit ODE solvers.

Implements explicit Runge-Kutta methods:
- Euler (RK1, 1st order)
- Midpoint (RK2, 2nd order)
- RK4 (classical 4th order)
- DOPRI5 (Dormand-Prince 5th order with adaptive stepping)
"""

from typing import Tuple, Optional, Dict, List
import torch

from kinopulse.core import DynamicalSystem, LegacyState, LegacyTime, Parameters, create_state
from kinopulse.solvers.config import SolverConfig
from .base import ODESolver
from .adaptive import AdaptiveODESolver
from .tableau import ButcherTableau


def _evaluate_rk_stages(
    system: DynamicalSystem,
    t: float,
    x: LegacyState,
    dt: float,
    tableau: ButcherTableau,
    u: Optional[torch.Tensor] = None,
    params: Optional[Parameters] = None,
) -> torch.Tensor:
    """Evaluate Runge-Kutta stages using Butcher tableau.

    Computes all stage derivatives k_i for a Runge-Kutta method:

        k_i = f(t + c_i * dt, x + dt * sum(A[i,j] * k_j for j < i))

    Args:
        system: Dynamical system
        t: Current time
        x: Current state
        dt: Time step size
        tableau: Butcher tableau with coefficients
        u: Optional input
        params: Optional parameters

    Returns:
        Tensor of shape (*batch_shape, n_stages, state_dim) containing all k_i

    Example:
        >>> tableau = ButcherTableau.from_name('rk4')
        >>> k = _evaluate_rk_stages(system, t, x, dt, tableau)
        >>> # k has shape (batch, 4, state_dim) for RK4
    """
    # Convert state to tensor
    if isinstance(x, LegacyState):
        x_tensor = x.tensor
    else:
        x_tensor = x

    # Get dimensions
    batch_shape = x_tensor.shape[:-1]
    state_dim = x_tensor.shape[-1]
    n_stages = len(tableau.b)

    # Get device and dtype
    device = x_tensor.device
    dtype = x_tensor.dtype

    # Allocate storage for all stages
    # Shape: (*batch_shape, n_stages, state_dim)
    k = torch.zeros(*batch_shape, n_stages, state_dim, dtype=dtype, device=device)

    # Evaluate each stage
    for i in range(n_stages):
        # Compute stage input: x_i = x + dt * sum(A[i,j] * k[j] for j < i)
        x_i = x_tensor.clone()

        for j in range(i):
            # Add contribution from previous stages
            # x_i += dt * A[i,j] * k[..., j, :]
            x_i = x_i + dt * tableau.A[i, j] * k[..., j, :]

        # Create State object for dynamics evaluation
        x_i_state = create_state(x_i)

        # Compute time for this stage
        t_i = t + tableau.c[i] * dt

        # Evaluate dynamics at stage point
        k_i = system.dynamics(LegacyTime(t_i), x_i_state, u, params)

        # Convert to tensor and store
        if isinstance(k_i, LegacyState):
            k_i_tensor = k_i.tensor
        else:
            k_i_tensor = k_i

        k[..., i, :] = k_i_tensor

    return k


def _combine_rk_stages(
    x: torch.Tensor,
    k: torch.Tensor,
    dt: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Combine RK stages using weights to compute final solution.

    Computes: x_new = x + dt * sum(weights[i] * k[i] for all i)

    Uses torch.einsum for efficient vectorized computation.

    Args:
        x: Current state tensor (*batch_shape, state_dim)
        k: Stage derivatives (*batch_shape, n_stages, state_dim)
        dt: Time step size
        weights: Stage weights (n_stages,) - either b or b_star from tableau

    Returns:
        New state x_new (*batch_shape, state_dim)

    Example:
        >>> x = torch.tensor([1.0, 2.0])
        >>> k = torch.randn(4, 2)  # 4 stages, 2D state
        >>> weights = torch.tensor([1/6, 2/6, 2/6, 1/6])  # RK4 weights
        >>> x_new = _combine_rk_stages(x, k, dt=0.1, weights=weights)
    """
    # Use einsum for efficient combination: sum over stages
    # '...sd,s->...d' means: for each batch (...), sum over stages (s), keep state dim (d)
    weighted_sum = torch.einsum('...sd,s->...d', k, weights)

    # Final update
    x_new = x + dt * weighted_sum

    return x_new


class EulerSolver(ODESolver):
    """Forward Euler solver (1st order).

    Integration formula:
        x_{n+1} = x_n + dt * f(t_n, x_n)

    Properties:
    - Order: 1
    - Stages: 1
    - Stability: Conditionally stable (small dt required)
    - Best for: Non-stiff problems, quick prototyping

    Example:
        >>> from kinopulse.solvers import SolverConfig
        >>> from kinopulse.solvers.ode import EulerSolver
        >>> config = SolverConfig(step_size={'dt': 0.01})
        >>> solver = EulerSolver(config)
        >>> trajectory = solver.solve(system, (0, 10), x0)
    """

    def step(
        self,
        system: DynamicalSystem,
        t: float,
        x: LegacyState,
        dt: float,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> Tuple[LegacyState, Dict]:
        """Take single Euler step.

        Args:
            system: Dynamical system
            t: Current time
            x: Current state
            dt: Time step size
            u: Optional input
            params: Optional parameters

        Returns:
            Tuple of (x_new, metadata)
        """
        # Evaluate dynamics: f(t, x)
        dx = system.dynamics(LegacyTime(t), x, u, params)

        # Convert to tensor
        x_tensor = self._to_tensor(x)
        dx_tensor = self._to_tensor(dx)

        # Forward Euler update: x_new = x + dt * dx
        x_new_tensor = x_tensor + dt * dx_tensor

        # Wrap in State
        x_new = create_state(x_new_tensor)

        # Metadata
        metadata = {
            'error_estimate': 0.0,  # No error estimate for Euler
            'stages': 1,
            'method': 'euler',
        }

        return x_new, metadata


class MidpointSolver(ODESolver):
    """Midpoint method solver (RK2, 2nd order).

    Integration formula:
        k1 = f(t_n, x_n)
        k2 = f(t_n + dt/2, x_n + dt/2 * k1)
        x_{n+1} = x_n + dt * k2

    Properties:
    - Order: 2
    - Stages: 2
    - Stability: Better than Euler for same dt
    - Best for: Moderate accuracy, non-stiff problems

    Example:
        >>> from kinopulse.solvers import SolverConfig
        >>> from kinopulse.solvers.ode import MidpointSolver
        >>> config = SolverConfig(step_size={'dt': 0.01})
        >>> solver = MidpointSolver(config)
        >>> trajectory = solver.solve(system, (0, 10), x0)
    """

    def step(
        self,
        system: DynamicalSystem,
        t: float,
        x: LegacyState,
        dt: float,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> Tuple[LegacyState, Dict]:
        """Take single midpoint (RK2) step.

        Args:
            system: Dynamical system
            t: Current time
            x: Current state
            dt: Time step size
            u: Optional input
            params: Optional parameters

        Returns:
            Tuple of (x_new, metadata)
        """
        # Convert state to tensor
        x_tensor = self._to_tensor(x)

        # Stage 1: Evaluate at current point
        # k1 = f(t, x)
        k1 = system.dynamics(LegacyTime(t), x, u, params)
        k1_tensor = self._to_tensor(k1)

        # Compute midpoint state
        # x_mid = x + (dt/2) * k1
        x_mid_tensor = x_tensor + 0.5 * dt * k1_tensor
        x_mid = create_state(x_mid_tensor)

        # Stage 2: Evaluate at midpoint
        # k2 = f(t + dt/2, x_mid)
        t_mid = t + 0.5 * dt
        k2 = system.dynamics(LegacyTime(t_mid), x_mid, u, params)
        k2_tensor = self._to_tensor(k2)

        # Final update using midpoint derivative
        # x_new = x + dt * k2
        x_new_tensor = x_tensor + dt * k2_tensor

        # Wrap in State
        x_new = create_state(x_new_tensor)

        # Metadata
        metadata = {
            'error_estimate': 0.0,  # No embedded error estimate
            'stages': 2,
            'method': 'midpoint',
        }

        return x_new, metadata


class RK4Solver(ODESolver):
    """Classical 4th-order Runge-Kutta solver (RK4).

    Integration formula:
        k1 = f(t_n, x_n)
        k2 = f(t_n + dt/2, x_n + dt/2 * k1)
        k3 = f(t_n + dt/2, x_n + dt/2 * k2)
        k4 = f(t_n + dt, x_n + dt * k3)
        x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Properties:
    - Order: 4
    - Stages: 4
    - Stability: Better stability than lower-order methods
    - Best for: High accuracy, non-stiff problems, general use

    The classical RK4 method is the workhorse of ODE integration,
    providing a good balance of accuracy and computational cost.

    Example:
        >>> from kinopulse.solvers import SolverConfig
        >>> from kinopulse.solvers.ode import RK4Solver
        >>> config = SolverConfig(step_size={'dt': 0.01})
        >>> solver = RK4Solver(config)
        >>> trajectory = solver.solve(system, (0, 10), x0)
    """

    def step(
        self,
        system: DynamicalSystem,
        t: float,
        x: LegacyState,
        dt: float,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> Tuple[LegacyState, Dict]:
        """Take single RK4 step.

        Args:
            system: Dynamical system
            t: Current time
            x: Current state
            dt: Time step size
            u: Optional input
            params: Optional parameters

        Returns:
            Tuple of (x_new, metadata)
        """
        # Convert state to tensor
        x_tensor = self._to_tensor(x)

        # Stage 1: Evaluate at current point
        # k1 = f(t, x)
        k1 = system.dynamics(LegacyTime(t), x, u, params)
        k1_tensor = self._to_tensor(k1)

        # Stage 2: Evaluate at t + dt/2 using k1
        # x2 = x + (dt/2) * k1
        x2_tensor = x_tensor + 0.5 * dt * k1_tensor
        x2 = create_state(x2_tensor)
        k2 = system.dynamics(LegacyTime(t + 0.5 * dt), x2, u, params)
        k2_tensor = self._to_tensor(k2)

        # Stage 3: Evaluate at t + dt/2 using k2
        # x3 = x + (dt/2) * k2
        x3_tensor = x_tensor + 0.5 * dt * k2_tensor
        x3 = create_state(x3_tensor)
        k3 = system.dynamics(LegacyTime(t + 0.5 * dt), x3, u, params)
        k3_tensor = self._to_tensor(k3)

        # Stage 4: Evaluate at t + dt using k3
        # x4 = x + dt * k3
        x4_tensor = x_tensor + dt * k3_tensor
        x4 = create_state(x4_tensor)
        k4 = system.dynamics(LegacyTime(t + dt), x4, u, params)
        k4_tensor = self._to_tensor(k4)

        # Final update: weighted average of slopes
        # x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        x_new_tensor = x_tensor + (dt / 6.0) * (
            k1_tensor + 2.0 * k2_tensor + 2.0 * k3_tensor + k4_tensor
        )

        # Wrap in State
        x_new = create_state(x_new_tensor)

        # Metadata
        metadata = {
            'error_estimate': 0.0,  # No embedded error estimate
            'stages': 4,
            'method': 'rk4',
        }

        return x_new, metadata


class DOPRI5Solver(AdaptiveODESolver):
    """Dormand-Prince 5(4) adaptive solver.

    A 5th-order Runge-Kutta method with embedded 4th-order error estimation.
    Uses 7 stages per step with FSAL (First Same As Last) property.

    The method computes two solutions:
    - 5th order (used as the solution)
    - 4th order (used for error estimation)

    Error estimate = ||x5 - x4||

    Properties:
    - Order: 5 (local error O(dt^6), global error O(dt^5))
    - Embedded order: 4 (for error estimation)
    - Stages: 7 (but only 6 evaluations due to FSAL)
    - Adaptive: Yes (automatic step size control)
    - Best for: General purpose, high accuracy, automatic stepping

    The DOPRI5 method is the default in many numerical libraries
    (MATLAB ode45, scipy solve_ivp with 'RK45', etc.)

    FSAL Optimization:
    The last stage k7 of step n equals k1 of step n+1. This implementation
    caches k7 and reuses it, reducing dynamics evaluations from 7 to 6
    per step (after the first step).

    References:
        Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
        formulae", Journal of Computational and Applied Mathematics.

    Example:
        >>> from kinopulse.solvers import SolverConfig, ToleranceConfig
        >>> from kinopulse.solvers.ode import DOPRI5Solver
        >>> config = SolverConfig(
        ...     step_size={'dt': 0.1, 'adaptive': True},
        ...     tolerance=ToleranceConfig(rtol=1e-6, atol=1e-9)
        ... )
        >>> solver = DOPRI5Solver(config)
        >>> trajectory = solver.solve(system, (0, 10), x0)
    """

    # Dormand-Prince coefficients
    # Butcher tableau coefficients
    A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]

    # Time coefficients
    C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]

    # 5th order solution weights
    B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

    # 4th order solution weights (for error estimation)
    B_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize DOPRI5 solver.

        Args:
            config: Solver configuration (creates default if None)
        """
        super().__init__(config)
        self.order = 5  # For step size controller
        # FSAL cache: stores k7 from previous accepted step
        self._fsal_k: Optional[torch.Tensor] = None
        self._fsal_valid = False

    def invalidate_fsal(self):
        """Invalidate FSAL cache (call after rejected step or event)."""
        self._fsal_valid = False
        self._fsal_k = None

    def step(
        self,
        system: DynamicalSystem,
        t: float,
        x: LegacyState,
        dt: float,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> Tuple[LegacyState, Dict]:
        """Take single DOPRI5 step with error estimation.

        Uses FSAL optimization: if k7 from previous step is cached,
        reuses it as k1, saving one dynamics evaluation.

        Args:
            system: Dynamical system
            t: Current time
            x: Current state
            dt: Time step size
            u: Optional input
            params: Optional parameters

        Returns:
            Tuple of (x_new, metadata) where metadata includes error_estimate
        """
        # Convert state to tensor
        x_tensor = self._to_tensor(x)

        # Compute stages
        k = []

        # Stage 1: k1 = f(t, x)
        # Use FSAL cache if valid, otherwise compute
        if self._fsal_valid and self._fsal_k is not None:
            k.append(self._fsal_k)
        else:
            k1 = system.dynamics(LegacyTime(t), x, u, params)
            k.append(self._to_tensor(k1))

        # Stages 2-7
        for i in range(1, 7):
            # Compute intermediate state
            x_i = x_tensor
            for j, a_ij in enumerate(self.A[i]):
                # Ensure k[j] is tensor (fix for mixed State/tensor issue)
                k_j = self._to_tensor(k[j])
                x_i = x_i + dt * a_ij * k_j

            x_i_state = create_state(x_i)
            t_i = t + self.C[i] * dt

            # Evaluate dynamics at intermediate state
            k_i = system.dynamics(LegacyTime(t_i), x_i_state, u, params)
            k.append(self._to_tensor(k_i))

        # Compute 5th order solution
        x5_tensor = x_tensor
        for i, b_i in enumerate(self.B):
            k_i = self._to_tensor(k[i])
            x5_tensor = x5_tensor + dt * b_i * k_i

        # Compute 4th order solution for error estimation
        x4_tensor = x_tensor
        for i, b_star_i in enumerate(self.B_star):
            k_i = self._to_tensor(k[i])
            x4_tensor = x4_tensor + dt * b_star_i * k_i

        # Cache k7 for FSAL (k7 at x_new equals k1 at next step)
        # Note: k[6] is k7, evaluated at (t + dt, x5)
        self._fsal_k = k[6]
        self._fsal_valid = True

        # Error estimate: difference between 5th and 4th order solutions
        error_estimate_tensor = torch.abs(x5_tensor - x4_tensor)

        # Use 5th order solution
        x_new = create_state(x5_tensor)

        # Metadata
        metadata = {
            'error_estimate': error_estimate_tensor,
            'stages': 7,
            'method': 'dopri5',
            'order': 5,
            'embedded_order': 4,
            'fsal_used': self._fsal_valid,
        }

        return x_new, metadata


class Tsit5Solver(AdaptiveODESolver):
    """Tsitouras 5(4) adaptive solver.

    An optimized 5th-order Runge-Kutta method with embedded 4th-order error estimation.
    Uses 7 stages per step, optimized for low error constants and efficiency.

    The method computes two solutions:
    - 5th order (used as the solution)
    - 4th order (used for error estimation)

    Error estimate = ||x5 - x4||

    Properties:
    - Order: 5 (local error O(dt^6), global error O(dt^5))
    - Embedded order: 4 (for error estimation)
    - Stages: 7
    - Adaptive: Yes (automatic step size control)
    - Best for: General purpose, often more efficient than DOPRI5

    Tsitouras' method is optimized for low error constants, making it
    particularly efficient for non-stiff problems. It's the default in
    Julia's DifferentialEquations.jl as 'Tsit5'.

    References:
        Tsitouras, Ch. (2011), "Runge-Kutta pairs of order 5(4) satisfying
        only the first column simplifying assumption", Computers & Mathematics
        with Applications.

    Example:
        >>> from kinopulse.solvers import SolverConfig, ToleranceConfig
        >>> from kinopulse.solvers.ode import Tsit5Solver
        >>> config = SolverConfig(
        ...     step_size={'dt': 0.1, 'adaptive': True},
        ...     tolerance=ToleranceConfig(rtol=1e-6, atol=1e-9)
        ... )
        >>> solver = Tsit5Solver(config)
        >>> trajectory = solver.solve(system, (0, 10), x0)
    """

    # Tsitouras 5(4) coefficients
    # c nodes
    C = [
        0.0,
        0.161,
        0.327,
        0.9,
        0.9800255409045097,
        1.0,
        1.0
    ]

    # A matrix (lower triangular)
    A = [
        [],
        [0.161],
        [-0.008480655492356989, 0.335480655492357],
        [2.8971530571054935, -6.359448489975075, 4.3622954328695815],
        [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525],
        [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383],
        [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774]
    ]

    # 5th order solution weights (b)
    B = [
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0
    ]

    # 4th order embedded solution weights (b_star) for error estimation
    B_star = [
        0.001780011052226,
        0.000816434459657,
        -0.007880878010262,
        0.144711007173263,
        -0.582357165452555,
        0.458082105929187,
        1.0/66.0
    ]

    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize Tsit5 solver.

        Args:
            config: Solver configuration (creates default if None)
        """
        super().__init__(config)
        self.order = 5  # For step size controller

    def step(
        self,
        system: DynamicalSystem,
        t: float,
        x: LegacyState,
        dt: float,
        u: Optional[torch.Tensor] = None,
        params: Optional[Parameters] = None,
    ) -> Tuple[LegacyState, Dict]:
        """Take single Tsit5 step with error estimation.

        Args:
            system: Dynamical system
            t: Current time
            x: Current state
            dt: Time step size
            u: Optional input
            params: Optional parameters

        Returns:
            Tuple of (x_new, metadata) where metadata includes error_estimate
        """
        # Convert state to tensor
        x_tensor = self._to_tensor(x)

        # Compute stages
        k = []

        # Stage 1: k1 = f(t, x)
        k1 = system.dynamics(LegacyTime(t), x, u, params)
        k.append(self._to_tensor(k1))

        # Stages 2-7
        for i in range(1, 7):
            # Compute intermediate state
            x_i = x_tensor
            for j, a_ij in enumerate(self.A[i]):
                # Ensure k[j] is tensor (fix for mixed State/tensor issue)
                k_j = self._to_tensor(k[j])
                x_i = x_i + dt * a_ij * k_j

            x_i_state = create_state(x_i)
            t_i = t + self.C[i] * dt

            # Evaluate dynamics at intermediate state
            k_i = system.dynamics(LegacyTime(t_i), x_i_state, u, params)
            k.append(self._to_tensor(k_i))

        # Compute 5th order solution
        x5_tensor = x_tensor
        for i, b_i in enumerate(self.B):
            k_i = self._to_tensor(k[i])
            x5_tensor = x5_tensor + dt * b_i * k_i

        # Compute 4th order solution for error estimation
        x4_tensor = x_tensor
        for i, b_star_i in enumerate(self.B_star):
            k_i = self._to_tensor(k[i])
            x4_tensor = x4_tensor + dt * b_star_i * k_i

        # Error estimate: difference between 5th and 4th order solutions
        error_estimate_tensor = torch.abs(x5_tensor - x4_tensor)

        # Use 5th order solution
        x_new = create_state(x5_tensor)

        # Metadata
        metadata = {
            'error_estimate': error_estimate_tensor,
            'stages': 7,
            'method': 'tsit5',
            'order': 5,
            'embedded_order': 4,
        }

        return x_new, metadata
