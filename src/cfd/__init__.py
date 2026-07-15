"""Basic CFD data structures for mesh-backed scalar and vector fields."""

from .boundary_conditions import BoundaryCondition, FixedValueBC, ZeroGradientBC
from .diffusion_solver import assemble_diffusion_system, build_structured_mesh, solve_diffusion
from .equation import Equation
from .field import ScalarField, VectorField
from .linear_system import LinearSystem, SparseMatrix
from .mesh import Mesh
from .operators import divergence, gradient, laplacian
from .solvers import GaussSeidelSolver, JacobiSolver

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
    "SparseMatrix",
    "LinearSystem",
    "JacobiSolver",
    "GaussSeidelSolver",
    "build_structured_mesh",
    "assemble_diffusion_system",
    "solve_diffusion",
]
