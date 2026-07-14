"""Basic CFD data structures for mesh-backed scalar and vector fields."""

from .boundary_conditions import BoundaryCondition, FixedValueBC, ZeroGradientBC
from .equation import Equation
from .field import ScalarField, VectorField
from .mesh import Mesh
from .operators import divergence, gradient, laplacian

__all__ = [
    "Mesh",
    "ScalarField",
    "VectorField",
    "BoundaryCondition",
    "FixedValueBC",
    "ZeroGradientBC",
    "gradient",
    "laplacian",
    "divergence",
    "Equation",
]
