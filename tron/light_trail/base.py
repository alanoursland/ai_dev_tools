"""
Base classes and protocols for the export module.

This module defines the foundational abstractions for exporting dynamical
systems to various formats.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from kinopulse.core.system import DynamicalSystem
    from kinopulse.export.metadata import ExportMetadata


class ExportFormat(Enum):
    """Available export formats for dynamical systems.

    Attributes:
        TORCHSCRIPT: TorchScript JIT-compiled format for deployment
        ONNX: Open Neural Network Exchange format for cross-platform deployment
        LATEX: LaTeX format for publication-ready equations
        TEXT: Plain text format for human-readable equations
        AST: Abstract syntax tree format for structured representation
        JSON_GRAPH: JSON-based computation graph format
        GRAPHML: GraphML XML format for graph visualization tools
        DOT: Graphviz DOT format for graph visualization
        MATHEMATICA: Wolfram Mathematica format
        MATLAB: MATLAB Symbolic Math Toolbox format
    """

    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    LATEX = "latex"
    TEXT = "text"
    AST = "ast"
    JSON_GRAPH = "json_graph"
    GRAPHML = "graphml"
    DOT = "dot"
    MATHEMATICA = "mathematica"
    MATLAB = "matlab"


@dataclass
class ValidationResult:
    """Results from export validation.

    This dataclass contains the results of validating an exported artifact
    against the original system, including numerical error statistics and
    diagnostic information.

    Attributes:
        passed: Whether validation passed (all checks successful)
        max_error: Maximum absolute error across all test cases
        mean_error: Mean absolute error across all test cases
        shape_mismatch: Whether exported output has incorrect shape
        type_mismatch: Whether exported output has incorrect type
        error_details: Optional detailed error message for debugging
        test_cases: Number of test cases evaluated during validation
    """

    passed: bool
    max_error: float
    mean_error: float
    shape_mismatch: bool = False
    type_mismatch: bool = False
    error_details: Optional[str] = None
    test_cases: int = 0

    def __str__(self) -> str:
        """Human-readable validation result summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Validation {status}",
            f"Test cases: {self.test_cases}",
            f"Max error: {self.max_error:.2e}",
            f"Mean error: {self.mean_error:.2e}",
        ]

        if self.shape_mismatch:
            lines.append("Shape mismatch detected")
        if self.type_mismatch:
            lines.append("Type mismatch detected")
        if self.error_details:
            lines.append(f"Details: {self.error_details}")

        return "\n".join(lines)


@runtime_checkable
class Exporter(Protocol):
    """Protocol defining the interface for format-specific exporters.

    All export implementations must satisfy this protocol to be registered
    with the export manager. The protocol defines two core methods: export
    and validate.
    """

    def export(
        self,
        system: 'DynamicalSystem',
        metadata: 'ExportMetadata',
        **kwargs: Any
    ) -> Any:
        """Export a dynamical system to the target format.

        Args:
            system: The dynamical system to export
            metadata: Export metadata containing system information
            **kwargs: Format-specific export options

        Returns:
            The exported artifact in the target format. The exact type
            depends on the export format (e.g., torch.jit.ScriptModule
            for TorchScript, str for LaTeX, dict for JSON).

        Raises:
            ExportError: If export fails for any reason
        """
        ...

    def validate(
        self,
        exported: Any,
        original: 'DynamicalSystem',
        tolerance: float = 1e-6
    ) -> ValidationResult:
        """Validate exported artifact against the original system.

        This method checks numerical equivalence between the exported
        artifact and the original system by evaluating both on random
        test points and comparing outputs.

        Args:
            exported: The exported artifact to validate
            original: The original dynamical system
            tolerance: Maximum allowed absolute error for validation to pass

        Returns:
            ValidationResult containing error statistics and pass/fail status

        Raises:
            ValidationError: If validation cannot be performed
        """
        ...
