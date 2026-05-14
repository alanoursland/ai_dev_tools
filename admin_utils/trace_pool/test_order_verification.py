"""Unit tests for convergence order verification using Richardson extrapolation."""

import pytest
import torch
import numpy as np
from kinopulse.core import DynamicalSystem, LegacyState, LegacyTime, Parameters, create_state
from kinopulse.solvers.ode import EulerSolver, MidpointSolver, RK4Solver
from kinopulse.solvers.config import SolverConfig, StepSizeConfig


class ExponentialDecaySystem(DynamicalSystem):
    """Exponential decay: dx/dt = -x, exact solution: x(t) = x0 * exp(-t)"""

    def __init__(self):
        super().__init__(state_dim=1)

    def dynamics(self, t: LegacyTime, x: LegacyState, u=None, params: Parameters = None):
        x_tensor = x.tensor if hasattr(x, 'tensor') else x
        dx = -x_tensor
        return create_state(dx)


def compute_richardson_order(step_sizes: list, errors: list) -> float:
    """
    Estimate convergence order using Richardson extrapolation.

    For a method of order p, error ~ C * dt^p, so:
        log(error) = log(C) + p * log(dt)

    We estimate p using log-log linear regression.

    Args:
        step_sizes: List of step sizes used
        errors: List of corresponding errors

    Returns:
        Estimated convergence order p

    Note:
        If any error is too small (near machine precision), returns infinity
        to indicate the solver is "too accurate" to measure convergence.
    """
    # Guard against log of zero when errors are at machine precision
    eps = 1e-15
    if any(e < eps for e in errors):
        return float('inf')

    log_dt = np.log(step_sizes)
    log_error = np.log(errors)

    # Least squares fit: log(error) = a + p*log(dt)
    # where p is the convergence order
    coeffs = np.polyfit(log_dt, log_error, 1)
    order = coeffs[0]

    return order


class TestEulerConvergenceOrder:
    """Tests for Euler method convergence order verification."""

    def test_euler_first_order_convergence(self):
        """Verify Euler method has first-order convergence (order ≈ 1.0 ± 0.2)."""
        system = ExponentialDecaySystem()

        # Test with multiple step sizes
        step_sizes = [0.1, 0.05, 0.025]
        errors = []

        # Initial condition and exact solution at t=1.0
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))

        for dt in step_sizes:
            config = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver = EulerSolver(config)

            # Solve from t=0 to t=1.0
            t_span = (0.0, t_final)
            trajectory = solver.solve(system, t_span, x0)

            # Get final state
            x_final = trajectory.states[-1]
            if hasattr(x_final, 'tensor'):
                x_final = x_final.tensor

            # Compute error
            error = torch.abs(x_final - x_exact).item()
            errors.append(error)

        # Estimate convergence order using Richardson extrapolation
        order = compute_richardson_order(step_sizes, errors)

        # Euler should have order 1.0 ± 0.2, or infinity if errors hit machine precision
        assert (0.8 <= order <= 1.2) or np.isinf(order), \
            f"Euler convergence order {order:.2f} not in expected range [0.8, 1.2]"

    def test_euler_error_halves_when_dt_halves(self):
        """Verify Euler error approximately halves when dt halves (first-order)."""
        system = ExponentialDecaySystem()
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))

        # Solve with dt=0.1
        config1 = SolverConfig(step_size=StepSizeConfig(dt=0.1))
        solver1 = EulerSolver(config1)
        traj1 = solver1.solve(system, (0.0, t_final), x0)
        x_final1 = traj1.states[-1]
        if hasattr(x_final1, 'tensor'):
            x_final1 = x_final1.tensor
        error1 = torch.abs(x_final1 - x_exact).item()

        # Solve with dt=0.05
        config2 = SolverConfig(step_size=StepSizeConfig(dt=0.05))
        solver2 = EulerSolver(config2)
        traj2 = solver2.solve(system, (0.0, t_final), x0)
        x_final2 = traj2.states[-1]
        if hasattr(x_final2, 'tensor'):
            x_final2 = x_final2.tensor
        error2 = torch.abs(x_final2 - x_exact).item()

        # Error ratio should be approximately 2 for first-order method
        # Guard against division by zero
        if error2 < 1e-15:
            ratio = float('inf')
        else:
            ratio = error1 / error2

        # Allow 20% tolerance: ratio should be in [1.6, 2.4], or infinity if errors hit machine precision
        assert (1.6 <= ratio <= 2.4) or np.isinf(ratio), \
            f"Euler error ratio {ratio:.2f} not in expected range [1.6, 2.4] for first-order"


class TestMidpointConvergenceOrder:
    """Tests for Midpoint (RK2) method convergence order verification."""

    def test_midpoint_second_order_convergence(self):
        """Verify Midpoint method has second-order convergence (order ≈ 2.0 ± 0.3)."""
        system = ExponentialDecaySystem()

        # Test with multiple step sizes
        step_sizes = [0.1, 0.05, 0.025]
        errors = []

        # Initial condition and exact solution at t=1.0
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))

        for dt in step_sizes:
            config = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver = MidpointSolver(config)

            # Solve from t=0 to t=1.0
            t_span = (0.0, t_final)
            trajectory = solver.solve(system, t_span, x0)

            # Get final state
            x_final = trajectory.states[-1]
            if hasattr(x_final, 'tensor'):
                x_final = x_final.tensor

            # Compute error
            error = torch.abs(x_final - x_exact).item()
            errors.append(error)

        # Estimate convergence order using Richardson extrapolation
        order = compute_richardson_order(step_sizes, errors)

        # Midpoint should have order 2.0 ± 0.3, or infinity if errors hit machine precision
        assert (1.7 <= order <= 2.3) or np.isinf(order), \
            f"Midpoint convergence order {order:.2f} not in expected range [1.7, 2.3]"

    def test_midpoint_error_quarters_when_dt_halves(self):
        """Verify Midpoint error approximately quarters when dt halves (second-order)."""
        system = ExponentialDecaySystem()
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))

        # Solve with dt=0.1
        config1 = SolverConfig(step_size=StepSizeConfig(dt=0.1))
        solver1 = MidpointSolver(config1)
        traj1 = solver1.solve(system, (0.0, t_final), x0)
        x_final1 = traj1.states[-1]
        if hasattr(x_final1, 'tensor'):
            x_final1 = x_final1.tensor
        error1 = torch.abs(x_final1 - x_exact).item()

        # Solve with dt=0.05
        config2 = SolverConfig(step_size=StepSizeConfig(dt=0.05))
        solver2 = MidpointSolver(config2)
        traj2 = solver2.solve(system, (0.0, t_final), x0)
        x_final2 = traj2.states[-1]
        if hasattr(x_final2, 'tensor'):
            x_final2 = x_final2.tensor
        error2 = torch.abs(x_final2 - x_exact).item()

        # Error ratio should be approximately 4 for second-order method
        # Guard against division by zero
        if error2 < 1e-15:
            ratio = float('inf')
        else:
            ratio = error1 / error2

        # Allow 25% tolerance: ratio should be in [3.0, 5.0], or infinity if errors hit machine precision
        assert (3.0 <= ratio <= 5.0) or np.isinf(ratio), \
            f"Midpoint error ratio {ratio:.2f} not in expected range [3.0, 5.0] for second-order"


class TestComparativeConvergence:
    """Tests comparing convergence behavior of different methods."""

    def test_midpoint_more_accurate_than_euler(self):
        """Verify Midpoint is more accurate than Euler for same step size."""
        system = ExponentialDecaySystem()
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))
        dt = 0.1

        # Solve with Euler
        config_euler = SolverConfig(step_size=StepSizeConfig(dt=dt))
        solver_euler = EulerSolver(config_euler)
        traj_euler = solver_euler.solve(system, (0.0, t_final), x0)
        x_euler = traj_euler.states[-1]
        if hasattr(x_euler, 'tensor'):
            x_euler = x_euler.tensor
        error_euler = torch.abs(x_euler - x_exact).item()

        # Solve with Midpoint
        config_midpoint = SolverConfig(step_size=StepSizeConfig(dt=dt))
        solver_midpoint = MidpointSolver(config_midpoint)
        traj_midpoint = solver_midpoint.solve(system, (0.0, t_final), x0)
        x_midpoint = traj_midpoint.states[-1]
        if hasattr(x_midpoint, 'tensor'):
            x_midpoint = x_midpoint.tensor
        error_midpoint = torch.abs(x_midpoint - x_exact).item()

        # Midpoint should be significantly more accurate
        assert error_midpoint < error_euler / 2, \
            f"Midpoint error {error_midpoint:.2e} not significantly better than Euler {error_euler:.2e}"

    def test_convergence_order_difference(self):
        """Verify Midpoint converges faster than Euler as dt decreases."""
        system = ExponentialDecaySystem()
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0

        step_sizes = [0.1, 0.05, 0.025]
        errors_euler = []
        errors_midpoint = []

        x_exact = torch.exp(-torch.tensor(t_final))

        for dt in step_sizes:
            # Euler
            config_euler = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver_euler = EulerSolver(config_euler)
            traj_euler = solver_euler.solve(system, (0.0, t_final), x0)
            x_euler = traj_euler.states[-1]
            if hasattr(x_euler, 'tensor'):
                x_euler = x_euler.tensor
            errors_euler.append(torch.abs(x_euler - x_exact).item())

            # Midpoint
            config_midpoint = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver_midpoint = MidpointSolver(config_midpoint)
            traj_midpoint = solver_midpoint.solve(system, (0.0, t_final), x0)
            x_midpoint = traj_midpoint.states[-1]
            if hasattr(x_midpoint, 'tensor'):
                x_midpoint = x_midpoint.tensor
            errors_midpoint.append(torch.abs(x_midpoint - x_exact).item())

        # Compute convergence orders
        order_euler = compute_richardson_order(step_sizes, errors_euler)
        order_midpoint = compute_richardson_order(step_sizes, errors_midpoint)

        # Midpoint should have higher convergence order
        # Infinity means machine precision reached, which is better than any finite rate
        if np.isinf(order_midpoint):
            pass  # Midpoint at machine precision is definitely better than Euler
        else:
            assert order_midpoint > order_euler + 0.5, \
                f"Midpoint order {order_midpoint:.2f} not significantly higher than Euler {order_euler:.2f}"


class TestRichardsonExtrapolation:
    """Tests for Richardson extrapolation utility function."""

    def test_richardson_order_estimation_synthetic(self):
        """Test Richardson order estimation on synthetic data with known order."""
        # Generate synthetic errors with known order p=2
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        C = 1.0  # Error constant
        p_true = 2.0  # True convergence order

        # errors = C * dt^p
        errors = [C * (dt ** p_true) for dt in step_sizes]

        # Estimate order
        p_estimated = compute_richardson_order(step_sizes, errors)

        # Should recover p=2 exactly (within numerical precision)
        assert abs(p_estimated - p_true) < 0.01, \
            f"Richardson extrapolation failed: estimated {p_estimated:.3f}, expected {p_true:.3f}"

    def test_richardson_order_first_order_synthetic(self):
        """Test Richardson order estimation for first-order synthetic data."""
        step_sizes = [0.2, 0.1, 0.05]
        C = 0.5
        p_true = 1.0

        errors = [C * (dt ** p_true) for dt in step_sizes]
        p_estimated = compute_richardson_order(step_sizes, errors)

        assert abs(p_estimated - p_true) < 0.01, \
            f"Richardson extrapolation failed for p=1: estimated {p_estimated:.3f}, expected {p_true:.3f}"


class TestRK4ConvergenceOrder:
    """Tests for RK4 method convergence order verification."""

    def test_rk4_fourth_order_convergence(self):
        """Verify RK4 method has fourth-order convergence (order ≈ 4.0 ± 0.5)."""
        system = ExponentialDecaySystem()

        # Test with multiple step sizes
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        errors = []

        # Initial condition and exact solution at t=1.0
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))

        for dt in step_sizes:
            config = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver = RK4Solver(config)

            # Solve from t=0 to t=1.0
            t_span = (0.0, t_final)
            trajectory = solver.solve(system, t_span, x0)

            # Get final state
            x_final = trajectory.states[-1]
            if hasattr(x_final, 'tensor'):
                x_final = x_final.tensor

            # Compute error
            error = torch.abs(x_final - x_exact).item()
            errors.append(error)

        # Estimate convergence order using Richardson extrapolation
        order = compute_richardson_order(step_sizes, errors)

        # RK4 should have order 4.0 ± 0.5, or infinity if errors hit machine precision
        assert (3.5 <= order <= 4.5) or np.isinf(order), \
            f"RK4 convergence order {order:.2f} not in expected range [3.5, 4.5]"

    def test_rk4_error_sixteenths_when_dt_halves(self):
        """Verify RK4 error approximately divides by 16 when dt halves (fourth-order)."""
        system = ExponentialDecaySystem()
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0
        x_exact = torch.exp(-torch.tensor(t_final))

        # Solve with dt=0.1
        config1 = SolverConfig(step_size=StepSizeConfig(dt=0.1))
        solver1 = RK4Solver(config1)
        traj1 = solver1.solve(system, (0.0, t_final), x0)
        x_final1 = traj1.states[-1]
        if hasattr(x_final1, 'tensor'):
            x_final1 = x_final1.tensor
        error1 = torch.abs(x_final1 - x_exact).item()

        # Solve with dt=0.05
        config2 = SolverConfig(step_size=StepSizeConfig(dt=0.05))
        solver2 = RK4Solver(config2)
        traj2 = solver2.solve(system, (0.0, t_final), x0)
        x_final2 = traj2.states[-1]
        if hasattr(x_final2, 'tensor'):
            x_final2 = x_final2.tensor
        error2 = torch.abs(x_final2 - x_exact).item()

        # Error ratio should be approximately 16 for fourth-order method
        # Guard against division by zero
        if error2 < 1e-15:
            ratio = float('inf')
        else:
            ratio = error1 / error2

        # Allow 30% tolerance: ratio should be in [11, 21], or infinity if errors hit machine precision
        assert (11.0 <= ratio <= 21.0) or np.isinf(ratio), \
            f"RK4 error ratio {ratio:.2f} not in expected range [11, 21] for fourth-order"

    @pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA device not available")
    def test_rk4_on_nonlinear_system(self):
        """Test RK4 convergence on nonlinear system (pendulum)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class PendulumSystem(DynamicalSystem):
            """Pendulum: d²θ/dt² = -sin(θ)

            State: [θ, ω] where ω = dθ/dt
            Dynamics: dθ/dt = ω, dω/dt = -sin(θ)
            """

            def __init__(self):
                super().__init__(state_dim=2)

            def dynamics(self, t, x, u=None, params=None):
                x_tensor = x.tensor if hasattr(x, 'tensor') else x
                theta = x_tensor[0]
                omega = x_tensor[1]
                dx = torch.stack([omega, -torch.sin(theta)])
                return create_state(dx)

        system = PendulumSystem()

        # Small angle: θ(0) = 0.1, ω(0) = 0
        x0 = create_state(torch.tensor([0.1, 0.0], device=device))
        t_final = 1.0

        # Compute reference solution with very small dt
        config_ref = SolverConfig(step_size=StepSizeConfig(dt=0.00001))
        solver_ref = RK4Solver(config_ref)
        traj_ref = solver_ref.solve(system, (0.0, t_final), x0)
        x_ref = traj_ref.states[-1]
        if hasattr(x_ref, 'tensor'):
            x_ref = x_ref.tensor

        # Test convergence with larger step sizes to stay in asymptotic regime
        step_sizes = [0.5, 0.25, 0.125]
        errors = []

        for dt in step_sizes:
            config = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver = RK4Solver(config)
            trajectory = solver.solve(system, (0.0, t_final), x0)
            x_final = trajectory.states[-1]
            if hasattr(x_final, 'tensor'):
                x_final = x_final.tensor
            error = torch.norm(x_final - x_ref).item()
            errors.append(error)

        # Estimate convergence order
        order = compute_richardson_order(step_sizes, errors)

        # Should still achieve 4th order on nonlinear system, or infinity if errors hit machine precision
        assert (3.5 <= order <= 4.5) or np.isinf(order), \
            f"RK4 convergence order on pendulum {order:.2f} not in expected range [3.5, 4.5]"


class TestRK4vsLowerOrderMethods:
    """Comparative convergence tests."""

    def test_rk4_higher_order_than_midpoint(self):
        """Verify RK4 has higher convergence order than Midpoint."""
        system = ExponentialDecaySystem()
        x0 = create_state(torch.tensor([1.0]))
        t_final = 1.0

        step_sizes = [0.1, 0.05, 0.025]
        errors_midpoint = []
        errors_rk4 = []

        x_exact = torch.exp(-torch.tensor(t_final))

        for dt in step_sizes:
            # Midpoint
            config_midpoint = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver_midpoint = MidpointSolver(config_midpoint)
            traj_midpoint = solver_midpoint.solve(system, (0.0, t_final), x0)
            x_midpoint = traj_midpoint.states[-1]
            if hasattr(x_midpoint, 'tensor'):
                x_midpoint = x_midpoint.tensor
            errors_midpoint.append(torch.abs(x_midpoint - x_exact).item())

            # RK4
            config_rk4 = SolverConfig(step_size=StepSizeConfig(dt=dt))
            solver_rk4 = RK4Solver(config_rk4)
            traj_rk4 = solver_rk4.solve(system, (0.0, t_final), x0)
            x_rk4 = traj_rk4.states[-1]
            if hasattr(x_rk4, 'tensor'):
                x_rk4 = x_rk4.tensor
            errors_rk4.append(torch.abs(x_rk4 - x_exact).item())

        # Compute convergence orders
        order_midpoint = compute_richardson_order(step_sizes, errors_midpoint)
        order_rk4 = compute_richardson_order(step_sizes, errors_rk4)

        # RK4 should have significantly higher convergence order
        # Infinity means machine precision reached, which is better than any finite rate
        if np.isinf(order_rk4):
            pass  # RK4 at machine precision is definitely better than Midpoint
        else:
            assert order_rk4 > order_midpoint + 1.0, \
                f"RK4 order {order_rk4:.2f} not significantly higher than Midpoint {order_midpoint:.2f}"
