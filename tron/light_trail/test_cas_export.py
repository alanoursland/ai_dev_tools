"""
Integration tests for external CAS exporters (Mathematica and MATLAB).

Tests cover:
- Full export workflows for Mathematica and MATLAB
- Complex symbolic systems (pendulum, Van der Pol, Lorenz)
- Cross-format consistency
- Jacobian template generation
- Parameter handling
"""

import pytest
from unittest.mock import Mock


class TestMathematicaFullExport:
    """Integration tests for complete Mathematica export."""

    def test_export_harmonic_oscillator(self):
        """Test exporting harmonic oscillator to Mathematica."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MathematicaExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Harmonic oscillator: x'' + x = 0
        x1, x2 = sp.symbols('x1 x2')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [x2, -x1]
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x1', 'x2']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="harmonic_oscillator",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x1', 'x2']
        )

        exporter = MathematicaExporter()
        mathematica = exporter.export(mock_system, metadata)

        # Validate
        result = exporter.validate(mathematica, mock_system)
        assert result.passed

        # Check content
        assert 'stateVars = {x1, x2}' in mathematica
        assert 'dynamics' in mathematica

    def test_export_van_der_pol(self):
        """Test exporting Van der Pol oscillator to Mathematica."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MathematicaExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Van der Pol: x'' - mu*(1-x^2)*x' + x = 0
        x, y, mu = sp.symbols('x y mu')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [y, mu*(1 - x**2)*y - x]
        mock_system.parameters = lambda: {'mu': 1.0}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x', 'y']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="van_der_pol",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x', 'y']
        )

        exporter = MathematicaExporter()
        mathematica = exporter.export(mock_system, metadata, include_parameters=True)

        # Validate
        result = exporter.validate(mathematica, mock_system)
        assert result.passed

        # Check for parameters
        assert 'mu = 1' in mathematica or 'mu = 1.0' in mathematica

    def test_export_lorenz_system(self):
        """Test exporting Lorenz system to Mathematica."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MathematicaExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Lorenz system
        x, y, z, sigma, rho, beta = sp.symbols('x y z sigma rho beta')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]
        mock_system.parameters = lambda: {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 3
        mock_system.metadata.state_names = ['x', 'y', 'z']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="lorenz",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=3,
            state_names=['x', 'y', 'z']
        )

        exporter = MathematicaExporter()
        mathematica = exporter.export(mock_system, metadata)

        # Validate
        result = exporter.validate(mathematica, mock_system)
        assert result.passed

        # Should have 3D state space
        assert 'stateVars = {x, y, z}' in mathematica

    def test_export_with_jacobian_workflow(self):
        """Test complete workflow with Jacobian computation."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MathematicaExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Pendulum system
        theta, omega, g, L = sp.symbols('theta omega g L')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [omega, -(g/L)*sp.sin(theta)]
        mock_system.parameters = lambda: {'g': 9.81, 'L': 1.0}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['theta', 'omega']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="pendulum",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['theta', 'omega']
        )

        exporter = MathematicaExporter()
        mathematica = exporter.export(
            mock_system,
            metadata,
            include_parameters=True,
            include_jacobian=True
        )

        # Validate
        result = exporter.validate(mathematica, mock_system)
        assert result.passed

        # Check for all components
        assert 'stateVars' in mathematica
        assert 'dynamics' in mathematica
        assert 'g = 9.81' in mathematica
        assert 'jacobianMatrix' in mathematica


class TestMATLABFullExport:
    """Integration tests for complete MATLAB export."""

    def test_export_harmonic_oscillator(self):
        """Test exporting harmonic oscillator to MATLAB."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MATLABExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Harmonic oscillator: x'' + x = 0
        x1, x2 = sp.symbols('x1 x2')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [x2, -x1]
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x1', 'x2']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="harmonic_oscillator",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x1', 'x2']
        )

        exporter = MATLABExporter()
        matlab = exporter.export(mock_system, metadata)

        # Validate
        result = exporter.validate(matlab, mock_system)
        assert result.passed

        # Check content
        assert 'syms x1 x2 real' in matlab
        assert 'f = [' in matlab

    def test_export_van_der_pol(self):
        """Test exporting Van der Pol oscillator to MATLAB."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MATLABExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Van der Pol: x'' - mu*(1-x^2)*x' + x = 0
        x, y, mu = sp.symbols('x y mu')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [y, mu*(1 - x**2)*y - x]
        mock_system.parameters = lambda: {'mu': 1.0}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x', 'y']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="van_der_pol",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x', 'y']
        )

        exporter = MATLABExporter()
        matlab = exporter.export(mock_system, metadata, include_parameters=True)

        # Validate
        result = exporter.validate(matlab, mock_system)
        assert result.passed

        # Check for parameters
        assert 'syms mu' in matlab
        assert 'mu = 1' in matlab or 'mu = 1.0' in matlab

    def test_export_lorenz_system(self):
        """Test exporting Lorenz system to MATLAB."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MATLABExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Lorenz system
        x, y, z, sigma, rho, beta = sp.symbols('x y z sigma rho beta')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]
        mock_system.parameters = lambda: {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 3
        mock_system.metadata.state_names = ['x', 'y', 'z']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="lorenz",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=3,
            state_names=['x', 'y', 'z']
        )

        exporter = MATLABExporter()
        matlab = exporter.export(mock_system, metadata)

        # Validate
        result = exporter.validate(matlab, mock_system)
        assert result.passed

        # Should have 3D state space
        assert 'syms x y z real' in matlab

    def test_export_with_jacobian_workflow(self):
        """Test complete workflow with Jacobian computation."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MATLABExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Pendulum system
        theta, omega, g, L = sp.symbols('theta omega g L')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [omega, -(g/L)*sp.sin(theta)]
        mock_system.parameters = lambda: {'g': 9.81, 'L': 1.0}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['theta', 'omega']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="pendulum",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['theta', 'omega']
        )

        exporter = MATLABExporter()
        matlab = exporter.export(
            mock_system,
            metadata,
            include_parameters=True,
            include_jacobian=True
        )

        # Validate
        result = exporter.validate(matlab, mock_system)
        assert result.passed

        # Check for all components
        assert 'syms theta omega' in matlab
        assert 'f = [' in matlab
        assert 'g = 9.81' in matlab
        assert 'A = jacobian(f, x_vec)' in matlab

    def test_export_function_format(self):
        """Test export as MATLAB function file."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import MATLABExporter
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        x1, x2 = sp.symbols('x1 x2')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [x2, -x1]
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x1', 'x2']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="harmonic_oscillator",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x1', 'x2']
        )

        exporter = MATLABExporter()
        matlab = exporter.export(mock_system, metadata, function_format=True)

        # Should be a function
        assert 'function [f, h, A] = harmonic_oscillator' in matlab
        assert 'end' in matlab

        # Still validate
        result = exporter.validate(matlab, mock_system)
        assert result.passed


class TestCrossFormatConsistency:
    """Test consistency across different export formats."""

    def test_same_system_different_formats(self):
        """Test that same system exports consistently across formats."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import (
                MathematicaExporter,
                MATLABExporter,
                LaTeXExporter,
                TextExporter
            )
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Simple system
        x, y = sp.symbols('x y')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [y, -x]
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x', 'y']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="test",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x', 'y']
        )

        # Export to all formats
        mathematica_exp = MathematicaExporter()
        matlab_exp = MATLABExporter()
        latex_exp = LaTeXExporter()
        text_exp = TextExporter()

        mathematica = mathematica_exp.export(mock_system, metadata)
        matlab = matlab_exp.export(mock_system, metadata)
        latex = latex_exp.export(mock_system, metadata)
        text = text_exp.export(mock_system, metadata)

        # All should validate
        assert mathematica_exp.validate(mathematica, mock_system).passed
        assert matlab_exp.validate(matlab, mock_system).passed
        assert latex_exp.validate(latex, mock_system).passed
        # TextExporter may not have validate method, skip if not present
        if hasattr(text_exp, 'validate'):
            assert text_exp.validate(text, mock_system).passed

        # All should contain the state variables
        assert 'x' in mathematica and 'y' in mathematica
        assert 'x' in matlab and 'y' in matlab
        assert 'x' in latex and 'y' in latex
        assert 'x' in text and 'y' in text

    def test_complex_system_all_formats(self):
        """Test complex system exports to all formats."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import (
                MathematicaExporter,
                MATLABExporter
            )
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        # Duffing oscillator with damping
        x, y, delta, alpha, beta, gamma, omega = sp.symbols(
            'x y delta alpha beta gamma omega'
        )
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [
            y,
            -delta*y - alpha*x - beta*x**3 + gamma*sp.cos(omega)
        ]
        mock_system.parameters = lambda: {
            'delta': 0.1,
            'alpha': -1.0,
            'beta': 1.0,
            'gamma': 0.3,
            'omega': 1.2
        }
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 2
        mock_system.metadata.state_names = ['x', 'y']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="duffing",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=2,
            state_names=['x', 'y']
        )

        # Export to Mathematica and MATLAB
        mathematica_exp = MathematicaExporter()
        matlab_exp = MATLABExporter()

        mathematica = mathematica_exp.export(
            mock_system,
            metadata,
            include_parameters=True,
            include_jacobian=True
        )
        matlab = matlab_exp.export(
            mock_system,
            metadata,
            include_parameters=True,
            include_jacobian=True
        )

        # Both should validate
        assert mathematica_exp.validate(mathematica, mock_system).passed
        assert matlab_exp.validate(matlab, mock_system).passed

        # Both should contain parameters
        for param in ['delta', 'alpha', 'beta', 'gamma']:
            assert param in mathematica
            assert param in matlab

        # Both should have Jacobian computation
        assert 'jacobian' in mathematica.lower()
        assert 'jacobian' in matlab.lower()

    def test_parameter_preservation_across_formats(self):
        """Test that parameters are preserved correctly across formats."""
        try:
            import sympy as sp
            from kinopulse.export.symbolic import (
                MathematicaExporter,
                MATLABExporter
            )
            from kinopulse.export.metadata import ExportMetadata
        except ImportError:
            pytest.skip("Required packages not available")

        x, m, k, c = sp.symbols('x m k c')
        mock_system = Mock()
        mock_system.symbolic_dynamics = lambda: [-(k/m)*x - (c/m)*x]
        mock_system.parameters = lambda: {'m': 1.0, 'k': 10.0, 'c': 0.5}
        mock_system.metadata = Mock()
        mock_system.metadata.state_dim = 1
        mock_system.metadata.state_names = ['x']
        mock_system.metadata.input_dim = None

        metadata = ExportMetadata(
            system_name="damped_oscillator",
            system_type="symbolic",
            kinopulse_version="0.1.0",
            state_dim=1,
            state_names=['x']
        )

        # Export with parameters
        mathematica_exp = MathematicaExporter()
        matlab_exp = MATLABExporter()

        mathematica = mathematica_exp.export(
            mock_system,
            metadata,
            include_parameters=True
        )
        matlab = matlab_exp.export(
            mock_system,
            metadata,
            include_parameters=True
        )

        # Check parameter values are correct in both
        assert 'm = 1' in mathematica or 'm = 1.0' in mathematica
        assert 'k = 10' in mathematica or 'k = 10.0' in mathematica
        assert 'c = 0.5' in mathematica

        assert 'm = 1' in matlab or 'm = 1.0' in matlab
        assert 'k = 10' in matlab or 'k = 10.0' in matlab
        assert 'c = 0.5' in matlab
