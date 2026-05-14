"""
State representations for dynamical systems.

Provides State class for representing system states with support for
flat vectors, hierarchical structures, and manifold constraints.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union
import torch

from kinopulse.utils import (
    to_device,
    to_dtype,
    get_device,
    get_dtype,
    clone_with_grad,
    InvalidStateError,
)


@dataclass
class StateSpec:
    """Specification for a state representation.

    Attributes:
        shape: State shape (batch_size, state_dim)
        dtype: Data type
        device: Device placement
        names: Optional semantic labels for state components
        constraints: Optional constraint functions
        manifold: Optional manifold structure
    """
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    names: Optional[list[str]] = None
    constraints: Optional[list[Callable]] = None
    manifold: Optional[Any] = None  # Manifold type deferred to M5

    def validate(self, x: torch.Tensor) -> bool:
        """Validate a tensor against this spec.

        Args:
            x: Tensor to validate

        Returns:
            True if valid, False otherwise
        """
        if x.shape != self.shape:
            return False
        if x.dtype != self.dtype:
            return False
        if x.device != self.device:
            return False

        # Check constraints
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(x):
                    return False

        return True


class State:
    """Dynamical system state representation.

    Supports flat vectors, hierarchical (dict/nested) states, and
    manifold-valued states with constraints.

    Example:
        >>> # Flat state
        >>> state = State(torch.randn(4, 10))  # batch_size=4, state_dim=10
        >>> state.shape
        torch.Size([4, 10])
        >>>
        >>> # Hierarchical state
        >>> state = State({'position': torch.randn(4, 3), 'velocity': torch.randn(4, 3)})
        >>> state['position'].shape
        torch.Size([4, 3])
    """

    def __init__(
        self,
        data: Union[torch.Tensor, Dict[str, 'State']],
        spec: Optional[StateSpec] = None,
        subsystem_states: Optional[Dict[str, 'State']] = None,
        mode: Optional[int] = None,
    ):
        """Initialize state.

        Args:
            data: State data (tensor or dict of substates)
            spec: Optional state specification
            subsystem_states: Optional subsystem states for hierarchical composition
            mode: Optional discrete mode (for hybrid systems)
        """
        self.spec = spec
        self.mode = mode

        if isinstance(data, torch.Tensor):
            # Flat tensor representation
            self._tensor = data
            self._subsystem_states = subsystem_states or {}
        elif isinstance(data, dict):
            # Hierarchical representation
            self._tensor = None
            self._subsystem_states = data
        else:
            raise TypeError(f"data must be torch.Tensor or dict, got {type(data)}")

    @property
    def tensor(self) -> torch.Tensor:
        """Get flat tensor representation.

        Returns:
            Flattened state tensor

        Raises:
            ValueError: If state is hierarchical and cannot be automatically flattened
        """
        if self._tensor is not None:
            return self._tensor

        # Try to flatten hierarchical state
        return self.flatten()

    @property
    def shape(self) -> torch.Size:
        """Get state shape."""
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype."""
        return self.tensor.dtype

    def to(self, device: torch.device) -> 'State':
        """Move state to device, preserving gradients.

        Args:
            device: Target device

        Returns:
            New state on target device
        """
        if self._tensor is not None:
            new_tensor = self._tensor.to(device)
            return State(
                new_tensor,
                spec=self.spec,
                subsystem_states={k: v.to(device) for k, v in self._subsystem_states.items()},
                mode=self.mode,
            )
        else:
            # Hierarchical state
            new_subsystems = {k: v.to(device) for k, v in self._subsystem_states.items()}
            return State(new_subsystems, spec=self.spec, mode=self.mode)

    def clone(self) -> 'State':
        """Deep copy with gradient preservation.

        Returns:
            Cloned state
        """
        if self._tensor is not None:
            new_tensor = clone_with_grad(self._tensor)
            return State(
                new_tensor,
                spec=self.spec,
                subsystem_states={k: v.clone() for k, v in self._subsystem_states.items()},
                mode=self.mode,
            )
        else:
            # Hierarchical state
            new_subsystems = {k: v.clone() for k, v in self._subsystem_states.items()}
            return State(new_subsystems, spec=self.spec, mode=self.mode)

    def validate(self) -> bool:
        """Check constraints and manifold membership.

        Returns:
            True if valid, False otherwise
        """
        if self.spec is None:
            return True

        return self.spec.validate(self.tensor)

    def project_to_manifold(self) -> 'State':
        """Project state to constraint manifold.

        Returns:
            Projected state

        Note:
            Requires StateSpec with manifold attribute set
        """
        if self.spec is None or self.spec.manifold is None:
            # No manifold constraint, return unchanged
            return self.clone()

        # Project tensor to manifold
        projected_tensor = self.spec.manifold.project(self.tensor)

        return State(
            projected_tensor,
            spec=self.spec,
            subsystem_states={k: v.clone() for k, v in self._subsystem_states.items()},
            mode=self.mode,
        )

    def tangent_at(self) -> 'TangentSpace':
        """Get tangent space at this state.

        Returns:
            Tangent space

        Note:
            Full implementation deferred to M5
        """
        # Placeholder - full implementation in M5
        raise NotImplementedError("Tangent space support deferred to M5")

    def __getitem__(self, key: Union[str, int]) -> 'State':
        """Hierarchical or component-wise access.

        Supports named subsystem lookup for hierarchical states and
        numeric access for flat tensor states (useful when tests refer
        to subsystems by index).

        Args:
            key: Subsystem name or component index

        Returns:
            Requested substate

        Raises:
            KeyError: If the requested entry cannot be found
        """
        # Direct hierarchical lookup
        normalized_key = str(key) if isinstance(key, int) else key
        if normalized_key in self._subsystem_states:
            return self._subsystem_states[normalized_key]

        # Fallback: allow numeric indexing into flat tensors
        if self._tensor is not None:
            if isinstance(key, str) and key.isdigit():
                key = int(key)

            if isinstance(key, int):
                if key < 0 or key >= self._tensor.shape[-1]:
                    raise KeyError(f"Subsystem '{key}' not found in state")

                component = self._tensor[..., key]
                if component.ndim == self._tensor.ndim - 1:
                    # Keep a consistent trailing dimension for state semantics
                    component = component.unsqueeze(-1)
                return State(component, spec=None, mode=self.mode)

        raise KeyError(f"Subsystem '{key}' not found in state")

    def flatten(self) -> torch.Tensor:
        """Flatten hierarchical state to single vector.

        Returns:
            Flattened state tensor
        """
        if self._tensor is not None:
            return self._tensor

        # Flatten hierarchical state
        if not self._subsystem_states:
            raise ValueError("Cannot flatten empty hierarchical state")

        tensors = []
        for key in sorted(self._subsystem_states.keys()):
            substate = self._subsystem_states[key]
            tensors.append(substate.flatten())

        return torch.cat(tensors, dim=-1)

    @classmethod
    def from_flat(cls, vec: torch.Tensor, spec: StateSpec) -> 'State':
        """Create state from flat vector.

        Args:
            vec: Flat state vector
            spec: State specification

        Returns:
            State instance
        """
        return cls(vec, spec=spec)


class TangentSpace:
    """Tangent space for manifold-valued states.

    Represents the tangent space at a point on a manifold, consisting of
    the base point and a basis for the tangent vectors.

    Attributes:
        base_point: Point on manifold where tangent space is evaluated
        basis: Optional basis vectors for the tangent space
        dimension: Dimension of the tangent space

    Example:
        >>> # Tangent space at identity rotation
        >>> base = torch.eye(3)
        >>> tangent = TangentSpace(base, dimension=3)
    """

    def __init__(
        self,
        base_point: torch.Tensor,
        basis: Optional[torch.Tensor] = None,
        dimension: Optional[int] = None,
    ):
        """Initialize tangent space.

        Args:
            base_point: Point on manifold
            basis: Optional basis vectors (shape: [..., dim, ambient_dim])
            dimension: Dimension of tangent space
        """
        self.base_point = base_point
        self.basis = basis
        self.dimension = dimension if dimension is not None else base_point.shape[-1]

    def project(self, v: torch.Tensor) -> torch.Tensor:
        """Project vector onto tangent space.

        Args:
            v: Vector to project

        Returns:
            Projected vector in tangent space
        """
        if self.basis is None:
            # No explicit basis, assume Euclidean
            return v

        # Project using basis (v_proj = sum_i <v, b_i> b_i)
        projections = torch.einsum('...i,...ji->...j', v, self.basis)
        return torch.einsum('...j,...ji->...i', projections, self.basis)


# ============================================================================
# Manifold Protocol and Implementations
# ============================================================================


from typing import Protocol


class Manifold(Protocol):
    """Protocol for manifold constraints on state spaces.

    Defines the interface for projections, tangent spaces, and
    exponential/logarithmic maps on manifolds.
    """

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project point to manifold.

        Args:
            x: Point (possibly off-manifold)

        Returns:
            Nearest point on manifold
        """
        ...

    def tangent_space(self, x: torch.Tensor) -> TangentSpace:
        """Get tangent space at point.

        Args:
            x: Point on manifold

        Returns:
            Tangent space at x
        """
        ...

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space to manifold.

        Args:
            x: Base point on manifold
            v: Tangent vector at x

        Returns:
            Point on manifold reached by following geodesic
        """
        ...

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from manifold to tangent space.

        Args:
            x: Base point on manifold
            y: Target point on manifold

        Returns:
            Tangent vector at x pointing toward y
        """
        ...


class SO3Manifold:
    """Special Orthogonal Group SO(3) - rotation matrices.

    Manifold of 3×3 rotation matrices with det(R) = +1 and R^T R = I.
    The tangent space at any point consists of skew-symmetric matrices.

    Example:
        >>> manifold = SO3Manifold()
        >>>
        >>> # Project arbitrary matrix to SO(3)
        >>> A = torch.randn(3, 3)
        >>> R = manifold.project(A)
        >>> print(torch.allclose(R.T @ R, torch.eye(3)))  # True
        >>>
        >>> # Exponential map
        >>> omega = torch.tensor([0.1, 0.2, 0.3])  # axis-angle
        >>> R_new = manifold.exp_map(torch.eye(3), omega)
    """

    def project(self, R: torch.Tensor) -> torch.Tensor:
        """Project matrix to SO(3) using SVD.

        Args:
            R: 3×3 matrix (or batch of matrices)

        Returns:
            Nearest rotation matrix
        """
        # Handle batched input
        original_shape = R.shape
        if R.ndim == 2:
            R = R.unsqueeze(0)  # Add batch dimension

        # Compute SVD: R = U S V^T
        U, _, Vh = torch.linalg.svd(R)

        # Closest rotation: R_proj = U V^T
        R_proj = U @ Vh

        # Ensure det(R) = +1 (not -1)
        det = torch.linalg.det(R_proj)

        # Fix reflections (det = -1)
        needs_fix = det < 0
        if needs_fix.any():
            # Flip sign of last column of U
            U_fixed = U.clone()
            U_fixed[needs_fix, :, -1] *= -1
            R_proj[needs_fix] = U_fixed[needs_fix] @ Vh[needs_fix]

        # Remove batch dimension if input was single matrix
        if len(original_shape) == 2:
            R_proj = R_proj.squeeze(0)

        return R_proj

    def tangent_space(self, R: torch.Tensor) -> TangentSpace:
        """Get tangent space at rotation R.

        The tangent space of SO(3) at any point consists of
        skew-symmetric 3×3 matrices (so(3) Lie algebra).

        Args:
            R: Rotation matrix

        Returns:
            Tangent space with basis of skew-symmetric matrices
        """
        # Basis for so(3): skew-symmetric matrices
        # e1 = [0, -1, 0; 1, 0, 0; 0, 0, 0] (rotation around z)
        # e2 = [0, 0, 1; 0, 0, 0; -1, 0, 0] (rotation around y)
        # e3 = [0, 0, 0; 0, 0, -1; 0, 1, 0] (rotation around x)

        device = R.device
        dtype = R.dtype

        basis = torch.zeros(3, 3, 3, device=device, dtype=dtype)
        # e1
        basis[0, 0, 1] = -1
        basis[0, 1, 0] = 1
        # e2
        basis[1, 0, 2] = 1
        basis[1, 2, 0] = -1
        # e3
        basis[2, 1, 2] = -1
        basis[2, 2, 1] = 1

        return TangentSpace(R, basis=basis, dimension=3)

    def exp_map(self, R: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Exponential map using Rodrigues' formula.

        Computes R_new = R @ exp(omega_hat) where omega_hat is the
        skew-symmetric matrix form of omega.

        Args:
            R: Base rotation matrix
            omega: Tangent vector (3D axis-angle representation)

        Returns:
            New rotation matrix
        """
        # Convert omega to skew-symmetric matrix
        omega_hat = self._skew_symmetric(omega)

        # Rodrigues' formula: exp(omega_hat) = I + sin(theta)/theta * omega_hat
        #                                        + (1-cos(theta))/theta^2 * omega_hat^2
        theta = torch.norm(omega, dim=-1, keepdim=True)

        # Handle small angles for numerical stability
        small_angle = theta < 1e-6

        # Compute coefficients
        if small_angle.any():
            # Taylor series for small angles
            coeff1 = torch.where(small_angle,
                                 torch.ones_like(theta) - theta**2 / 6,
                                 torch.sin(theta) / theta)
            coeff2 = torch.where(small_angle,
                                 0.5 * torch.ones_like(theta) - theta**2 / 24,
                                 (1 - torch.cos(theta)) / theta**2)
        else:
            coeff1 = torch.sin(theta) / theta
            coeff2 = (1 - torch.cos(theta)) / theta**2

        # Reshape for broadcasting
        if omega.ndim == 1:
            coeff1 = coeff1.item()
            coeff2 = coeff2.item()
        else:
            coeff1 = coeff1.squeeze(-1)
            coeff2 = coeff2.squeeze(-1)

        # exp(omega_hat) = I + coeff1 * omega_hat + coeff2 * omega_hat^2
        I = torch.eye(3, device=omega.device, dtype=omega.dtype)
        omega_hat_sq = omega_hat @ omega_hat

        exp_omega = I + coeff1 * omega_hat + coeff2 * omega_hat_sq

        # R_new = R @ exp(omega_hat)
        return R @ exp_omega

    def log_map(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from SO(3) to tangent space.

        Computes the tangent vector omega such that R2 ≈ R1 @ exp(omega_hat).

        Args:
            R1: Base rotation
            R2: Target rotation

        Returns:
            Tangent vector (axis-angle)
        """
        # Compute relative rotation: R_rel = R1^T @ R2
        R_rel = R1.T @ R2

        # Extract axis-angle from rotation matrix
        # theta = arccos((trace(R) - 1) / 2)
        trace = torch.trace(R_rel)
        theta = torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))

        # Handle small angles
        if theta.abs() < 1e-6:
            # Near identity - use first-order approximation
            omega_hat = (R_rel - R_rel.T) / 2
            omega = self._vee(omega_hat)
            return omega

        # Extract axis from skew-symmetric part
        # omega_hat = theta / (2 sin(theta)) * (R - R^T)
        coeff = theta / (2 * torch.sin(theta))
        omega_hat = coeff * (R_rel - R_rel.T)
        omega = self._vee(omega_hat)

        return omega

    @staticmethod
    def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """Convert 3-vector to 3×3 skew-symmetric matrix.

        Args:
            v: 3D vector

        Returns:
            3×3 skew-symmetric matrix
        """
        if v.ndim == 1:
            # Single vector
            v1, v2, v3 = v[0], v[1], v[2]
            return torch.tensor([
                [0, -v3, v2],
                [v3, 0, -v1],
                [-v2, v1, 0]
            ], device=v.device, dtype=v.dtype)
        else:
            # Batched vectors
            batch_shape = v.shape[:-1]
            result = torch.zeros(*batch_shape, 3, 3, device=v.device, dtype=v.dtype)
            result[..., 0, 1] = -v[..., 2]
            result[..., 0, 2] = v[..., 1]
            result[..., 1, 0] = v[..., 2]
            result[..., 1, 2] = -v[..., 0]
            result[..., 2, 0] = -v[..., 1]
            result[..., 2, 1] = v[..., 0]
            return result

    @staticmethod
    def _vee(omega_hat: torch.Tensor) -> torch.Tensor:
        """Extract 3-vector from skew-symmetric matrix (inverse of skew).

        Args:
            omega_hat: 3×3 skew-symmetric matrix

        Returns:
            3D vector
        """
        return torch.tensor([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0]
        ], device=omega_hat.device, dtype=omega_hat.dtype)


class SE3Manifold:
    """Special Euclidean Group SE(3) - rigid body transformations.

    Manifold of 4×4 homogeneous transformation matrices representing
    rotations and translations in 3D space.

    A transformation T has the form:
        T = [R  t]
            [0  1]
    where R ∈ SO(3) and t ∈ ℝ³.

    Example:
        >>> manifold = SE3Manifold()
        >>>
        >>> # Create transformation
        >>> R = torch.eye(3)
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>> T = torch.zeros(4, 4)
        >>> T[:3, :3] = R
        >>> T[:3, 3] = t
        >>> T[3, 3] = 1.0
        >>>
        >>> # Project to ensure it's on SE(3)
        >>> T_proj = manifold.project(T)
    """

    def __init__(self):
        """Initialize SE(3) manifold."""
        self.so3 = SO3Manifold()

    def project(self, T: torch.Tensor) -> torch.Tensor:
        """Project 4×4 matrix to SE(3).

        Projects the rotation part to SO(3) and keeps translation unchanged.

        Args:
            T: 4×4 transformation matrix

        Returns:
            Nearest SE(3) transformation
        """
        T_proj = T.clone()

        # Project rotation part to SO(3)
        R = T[..., :3, :3]
        R_proj = self.so3.project(R)
        T_proj[..., :3, :3] = R_proj

        # Ensure bottom row is [0, 0, 0, 1]
        T_proj[..., 3, :] = torch.tensor([0, 0, 0, 1],
                                         device=T.device, dtype=T.dtype)

        return T_proj

    def tangent_space(self, T: torch.Tensor) -> TangentSpace:
        """Get tangent space at transformation T.

        The tangent space of SE(3) is the se(3) Lie algebra.

        Args:
            T: Transformation matrix

        Returns:
            Tangent space
        """
        # se(3) has dimension 6 (3 for rotation, 3 for translation)
        return TangentSpace(T, dimension=6)

    def exp_map(self, T: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Exponential map for SE(3).

        Args:
            T: Base transformation
            xi: Tangent vector (6D: [omega, v] for rotation and translation)

        Returns:
            New transformation
        """
        # Split into rotation and translation parts
        omega = xi[..., :3]  # Rotation part (axis-angle)
        v = xi[..., 3:]      # Translation part

        # Compute rotation exponential
        R = T[..., :3, :3]
        R_new = self.so3.exp_map(R, omega)

        # Compute translation part (more complex for SE(3))
        theta = torch.norm(omega, dim=-1, keepdim=True)

        # For small rotations, V ≈ I
        small_angle = theta < 1e-6

        if small_angle.any():
            # Use approximation for small angles
            V = torch.eye(3, device=xi.device, dtype=xi.dtype)
        else:
            # V = I + (1-cos(theta))/theta^2 * omega_hat
            #       + (theta-sin(theta))/theta^3 * omega_hat^2
            omega_hat = self.so3._skew_symmetric(omega)
            omega_hat_sq = omega_hat @ omega_hat

            coeff1 = (1 - torch.cos(theta)) / theta**2
            coeff2 = (theta - torch.sin(theta)) / theta**3

            I = torch.eye(3, device=xi.device, dtype=xi.dtype)
            V = I + coeff1.squeeze(-1) * omega_hat + coeff2.squeeze(-1) * omega_hat_sq

        # Translation update
        t = T[..., :3, 3]
        t_new = t + V @ v

        # Construct new transformation
        T_new = torch.zeros_like(T)
        T_new[..., :3, :3] = R_new
        T_new[..., :3, 3] = t_new
        T_new[..., 3, 3] = 1.0

        return T_new

    def log_map(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """Logarithmic map for SE(3).

        Args:
            T1: Base transformation
            T2: Target transformation

        Returns:
            Tangent vector (6D)
        """
        # Compute relative transformation
        T_rel = torch.linalg.inv(T1) @ T2

        # Extract rotation and translation
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        # Compute rotation logarithm
        omega = self.so3.log_map(torch.eye(3, device=R_rel.device), R_rel)

        # Compute translation part (inverse of exponential)
        theta = torch.norm(omega)

        if theta < 1e-6:
            v = t_rel
        else:
            omega_hat = self.so3._skew_symmetric(omega)
            omega_hat_sq = omega_hat @ omega_hat

            coeff1 = (1 - torch.cos(theta)) / theta**2
            coeff2 = (theta - torch.sin(theta)) / theta**3

            I = torch.eye(3, device=omega.device, dtype=omega.dtype)
            V = I + coeff1 * omega_hat + coeff2 * omega_hat_sq

            v = torch.linalg.inv(V) @ t_rel

        # Combine into 6D tangent vector
        return torch.cat([omega, v])
