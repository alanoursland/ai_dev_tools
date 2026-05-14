"""
Export metadata for dynamical systems.

This module defines the metadata structure attached to all exported systems,
providing comprehensive information about system properties, dimensions,
parameters, and provenance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from kinopulse.export.base import ExportFormat


@dataclass
class ExportMetadata:
    """Comprehensive metadata for exported dynamical systems.

    This dataclass captures all relevant information about an exported system,
    including identification, dimensions, parameters, stability certificates,
    and provenance. The metadata is designed to be serializable and can be
    embedded in various export formats.

    Attributes:
        # System Identification
        system_name: Human-readable name of the system
        system_type: Type categorization (symbolic, numeric, neural, hybrid, pde, composite)
        kinopulse_version: Version of KinoPulse used for export

        # Dimensions and Structure
        state_dim: Dimension of the state space
        input_dim: Dimension of the input space (None if autonomous)
        output_dim: Dimension of the output space
        num_modes: Number of discrete modes (for hybrid systems)

        # State/Input/Output Semantics
        state_names: Optional list of state variable names
        input_names: Optional list of input variable names
        output_names: Optional list of output variable names

        # Parameter Information
        parameter_count: Total number of parameters in the system
        trainable_params: Number of trainable/learnable parameters

        # PDE-Specific Metadata
        spatial_dims: Number of spatial dimensions (for PDE systems)
        grid_shape: Shape of spatial discretization grid
        boundary_conditions: Dictionary describing boundary conditions

        # Stability Certificates
        lyapunov_function: Symbolic or textual representation of Lyapunov function
        roa_estimate: Region of attraction estimate data
        stability_margin: Numerical stability margin

        # Controller Metadata (for closed-loop exports)
        controller_type: Type of controller (e.g., "LQR", "MPC", "neural")
        design_point: Operating point used for controller design

        # Provenance
        export_timestamp: ISO timestamp of when export was created
        export_format: Format to which system was exported
        source_module: Source module that created the system

        # Custom Annotations
        annotations: Dictionary for arbitrary custom metadata
    """

    # System identification (required)
    system_name: str
    system_type: str  # "symbolic", "numeric", "neural", "hybrid", "pde", "composite"
    kinopulse_version: str

    # Dimensions and structure (required)
    state_dim: int

    # Dimensions and structure (optional)
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    num_modes: Optional[int] = None  # For hybrid systems

    # State/input/output semantics
    state_names: Optional[List[str]] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None

    # Parameter information
    parameter_count: int = 0
    trainable_params: int = 0

    # PDE-specific metadata
    spatial_dims: Optional[int] = None
    grid_shape: Optional[tuple] = None
    boundary_conditions: Optional[Dict[str, str]] = None

    # Stability certificates
    lyapunov_function: Optional[str] = None
    roa_estimate: Optional[Dict[str, Any]] = None
    stability_margin: Optional[float] = None

    # Controller metadata (for closed-loop exports)
    controller_type: Optional[str] = None
    design_point: Optional[Dict[str, float]] = None

    # Provenance
    export_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    export_format: Optional[ExportFormat] = None
    source_module: Optional[str] = None

    # Custom annotations
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary for serialization.

        Returns:
            Dictionary representation of all metadata fields. ExportFormat
            enums are converted to their string values.
        """
        data = {
            'system_name': self.system_name,
            'system_type': self.system_type,
            'kinopulse_version': self.kinopulse_version,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_modes': self.num_modes,
            'state_names': self.state_names,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'parameter_count': self.parameter_count,
            'trainable_params': self.trainable_params,
            'spatial_dims': self.spatial_dims,
            'grid_shape': self.grid_shape,
            'boundary_conditions': self.boundary_conditions,
            'lyapunov_function': self.lyapunov_function,
            'roa_estimate': self.roa_estimate,
            'stability_margin': self.stability_margin,
            'controller_type': self.controller_type,
            'design_point': self.design_point,
            'export_timestamp': self.export_timestamp,
            'export_format': self.export_format.value if self.export_format else None,
            'source_module': self.source_module,
            'annotations': self.annotations,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExportMetadata':
        """Create ExportMetadata from a dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            ExportMetadata instance populated from the dictionary

        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
        """
        # Convert export_format string back to enum
        export_format = data.get('export_format')
        if export_format is not None and isinstance(export_format, str):
            export_format = ExportFormat(export_format)

        # Convert grid_shape list back to tuple if present
        grid_shape = data.get('grid_shape')
        if grid_shape is not None and isinstance(grid_shape, list):
            grid_shape = tuple(grid_shape)

        return cls(
            system_name=data['system_name'],
            system_type=data['system_type'],
            kinopulse_version=data['kinopulse_version'],
            state_dim=data['state_dim'],
            input_dim=data.get('input_dim'),
            output_dim=data.get('output_dim'),
            num_modes=data.get('num_modes'),
            state_names=data.get('state_names'),
            input_names=data.get('input_names'),
            output_names=data.get('output_names'),
            parameter_count=data.get('parameter_count', 0),
            trainable_params=data.get('trainable_params', 0),
            spatial_dims=data.get('spatial_dims'),
            grid_shape=grid_shape,
            boundary_conditions=data.get('boundary_conditions'),
            lyapunov_function=data.get('lyapunov_function'),
            roa_estimate=data.get('roa_estimate'),
            stability_margin=data.get('stability_margin'),
            controller_type=data.get('controller_type'),
            design_point=data.get('design_point'),
            export_timestamp=data.get('export_timestamp', datetime.now().isoformat()),
            export_format=export_format,
            source_module=data.get('source_module'),
            annotations=data.get('annotations', {}),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"ExportMetadata: {self.system_name}",
            f"  Type: {self.system_type}",
            f"  State dim: {self.state_dim}",
        ]

        if self.input_dim is not None:
            lines.append(f"  Input dim: {self.input_dim}")
        if self.output_dim is not None:
            lines.append(f"  Output dim: {self.output_dim}")
        if self.num_modes is not None:
            lines.append(f"  Modes: {self.num_modes}")
        if self.parameter_count > 0:
            lines.append(f"  Parameters: {self.parameter_count} ({self.trainable_params} trainable)")
        if self.export_format:
            lines.append(f"  Format: {self.export_format.value}")

        return "\n".join(lines)
