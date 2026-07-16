"""Educational 2D Poisson equation solver.

This module extends the Sprint 18 diffusion solver by adding a source term,
turning Laplace's equation into the more general Poisson equation:

    d2(phi)/dx2 + d2(phi)/dy2 = source_term(x, y)      (i.e. Laplacian(phi) = f)

on a structured 2D Cartesian mesh with FixedValueBC (Dirichlet) boundaries on
all four sides. It reuses the same Mesh, ScalarField, SparseMatrix,
LinearSystem, Equation, and iterative solver infrastructure built in earlier
sprints rather than reimplementing any of it.

Nothing here implements transient diffusion, convection, pressure
correction, SIMPLE or Navier-Stokes — those are left for later sprints.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .equation import Equation
from .field import ScalarField
from .linear_system import LinearSystem, SparseMatrix
from .mesh import Mesh
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}

SourceTerm = Union[float, int, Callable[[float, float], float], np.ndarray, Sequence[float], ScalarField]


def poisson_equation() -> Equation:
    """Return an Equation describing the 2D Poisson problem this module solves.

    This is purely descriptive bookkeeping — it documents the PDE being
    assembled below using the same symbolic Equation container the rest of
    the CFD package uses, rather than encoding it as a bare string.

    Returns
    -------
    Equation
        ``laplacian(phi) = source_term``
    """
    equation = Equation(lhs="laplacian(phi)", rhs="source_term")
    return equation


def _validate_mesh_for_poisson(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencil used below assumes every interior point has a
    left/right/top/bottom neighbour at a constant spacing, so we verify that
    up front rather than failing confusingly deep inside assembly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("poisson_solver only supports 2D Cartesian meshes.")

    x_values = np.unique(np.round(mesh.cell_centers[:, 0], 12))
    y_values = np.unique(np.round(mesh.cell_centers[:, 1], 12))
    if x_values.size < 3 or y_values.size < 3:
        raise ValueError("mesh must contain at least three points along each axis.")

    dx = np.diff(x_values)
    dy = np.diff(y_values)
    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("mesh must be uniformly spaced (structured Cartesian) in each direction.")

    if mesh.n_cells != x_values.size * y_values.size:
        raise ValueError("mesh must have exactly one point at every (x, y) grid location.")


def _validate_boundary_conditions(boundary_conditions: Sequence[BoundaryCondition]) -> None:
    """Check that exactly one FixedValueBC is given for each of the four sides.

    poisson_solver only supports Dirichlet boundaries, so any other
    BoundaryCondition subclass — such as ZeroGradientBC — is rejected here
    rather than producing an incorrectly assembled system.
    """
    if not isinstance(boundary_conditions, (list, tuple)):
        raise TypeError("boundary_conditions must be a list or tuple of FixedValueBC instances.")
    if len(boundary_conditions) != 4:
        raise ValueError("exactly four boundary conditions are required: left, right, top, bottom.")

    seen = set()
    for bc in boundary_conditions:
        if not isinstance(bc, FixedValueBC):
            raise TypeError(
                f"poisson_solver only supports FixedValueBC (Dirichlet) boundaries, got {type(bc).__name__}."
            )
        if bc.boundary in seen:
            raise ValueError(f"boundary '{bc.boundary}' was specified more than once.")
        seen.add(bc.boundary)

    missing = _REQUIRED_BOUNDARIES - seen
    if missing:
        raise ValueError(f"missing boundary condition(s) for: {', '.join(sorted(missing))}")


def _boundary_cell_indices(mesh: Mesh, boundary: str) -> np.ndarray:
    """Return the indices of the mesh cells lying on the given boundary."""
    x, y = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]
    if boundary == "left":
        mask = np.isclose(x, x.min())
    elif boundary == "right":
        mask = np.isclose(x, x.max())
    elif boundary == "bottom":
        mask = np.isclose(y, y.min())
    else:  # "top"
        mask = np.isclose(y, y.max())
    return np.where(mask)[0]


def _evaluate_source_term(mesh: Mesh, source_term: SourceTerm) -> np.ndarray:
    """Turn any supported ``source_term`` representation into a per-cell array.

    Parameters
    ----------
    mesh : Mesh
        Mesh the source term must be evaluated on.
    source_term : number, callable, array-like, or ScalarField
        - A single number applies the same source everywhere.
        - A callable ``f(x, y)`` is evaluated at every cell center.
        - An array-like or ScalarField supplies one value per cell directly.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(mesh.n_cells,)`` with one source value per cell.
    """
    if isinstance(source_term, ScalarField):
        if source_term.mesh is not mesh:
            raise ValueError("source_term ScalarField must be attached to the same mesh being solved.")
        return source_term.values.copy()

    if isinstance(source_term, bool):
        raise TypeError("source_term must not be a bool.")

    if isinstance(source_term, (int, float)):
        return np.full(mesh.n_cells, float(source_term), dtype=float)

    if callable(source_term):
        return np.array([float(source_term(x, y)) for x, y in mesh.cell_centers], dtype=float)

    if isinstance(source_term, (list, tuple, np.ndarray)):
        values = np.asarray(source_term, dtype=float)
        if values.ndim != 1 or values.shape[0] != mesh.n_cells:
            raise ValueError(f"source_term array must have shape ({mesh.n_cells},).")
        return values

    raise TypeError(
        "source_term must be a number, a callable f(x, y), an array-like of length "
        "mesh.n_cells, or a ScalarField."
    )


def assemble_poisson_system(
    mesh: Mesh,
    source_term: SourceTerm,
    boundary_conditions: Sequence[BoundaryCondition],
) -> LinearSystem:
    """Assemble the linear system A*phi = b for Laplacian(phi) = source_term.

    Interior points get the standard five-point finite-difference stencil
    for the 2D Laplacian, with the source term evaluated at that point on the
    right-hand side:

        (phi[E] - 2*phi[P] + phi[W]) / dx^2
      + (phi[N] - 2*phi[P] + phi[S]) / dy^2 = source_term[P]

    Boundary points are handled by direct elimination: their matrix row is
    just ``1 * phi_boundary = fixed_value``, which pins that unknown to the
    FixedValueBC value instead of computing a stencil for it.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    source_term : number, callable, array-like, or ScalarField
        The right-hand side ``f`` of ``laplacian(phi) = f``. See
        :func:`_evaluate_source_term` for the accepted forms.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances, one per boundary
        ("left", "right", "top", "bottom").

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector.
    """
    _validate_mesh_for_poisson(mesh)
    _validate_boundary_conditions(boundary_conditions)
    source_values = _evaluate_source_term(mesh, source_term)

    coords = mesh.cell_centers
    x_coords = np.unique(np.round(coords[:, 0], 12))
    y_coords = np.unique(np.round(coords[:, 1], 12))
    dx = float(np.diff(x_coords)[0])
    dy = float(np.diff(y_coords)[0])
    nx, ny = x_coords.size, y_coords.size

    # Map (ix, iy) grid coordinates to a flat cell index, so we can look up
    # the E/W/N/S neighbours of any point regardless of how the mesh's
    # cell_centers happen to be ordered.
    grid_index = np.zeros((nx, ny), dtype=int)
    ix_of = np.zeros(mesh.n_cells, dtype=int)
    iy_of = np.zeros(mesh.n_cells, dtype=int)
    for cell_index, (x, y) in enumerate(coords):
        ix = int(np.where(np.isclose(x_coords, x))[0][0])
        iy = int(np.where(np.isclose(y_coords, y))[0][0])
        grid_index[ix, iy] = cell_index
        ix_of[cell_index] = ix
        iy_of[cell_index] = iy

    # Work out which cells sit on the boundary and what value each one
    # should be pinned to, using the FixedValueBC objects directly (rather
    # than duplicating their fixed values by hand).
    boundary_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    is_boundary = np.zeros(mesh.n_cells, dtype=bool)
    for bc in boundary_conditions:
        bc.apply(boundary_field)
        is_boundary[_boundary_cell_indices(mesh, bc.boundary)] = True

    n = mesh.n_cells
    matrix = SparseMatrix(size=n)
    rhs = np.zeros(n)

    coeff_x = 1.0 / dx ** 2
    coeff_y = 1.0 / dy ** 2

    for cell_index in range(n):
        if is_boundary[cell_index]:
            # Dirichlet elimination: phi at this cell is simply fixed.
            matrix.set(cell_index, cell_index, 1.0)
            rhs[cell_index] = boundary_field.values[cell_index]
            continue

        ix, iy = ix_of[cell_index], iy_of[cell_index]
        east = int(grid_index[ix + 1, iy])
        west = int(grid_index[ix - 1, iy])
        north = int(grid_index[ix, iy + 1])
        south = int(grid_index[ix, iy - 1])

        matrix.set(cell_index, cell_index, -2.0 * (coeff_x + coeff_y))
        matrix.set(cell_index, east, coeff_x)
        matrix.set(cell_index, west, coeff_x)
        matrix.set(cell_index, north, coeff_y)
        matrix.set(cell_index, south, coeff_y)
        rhs[cell_index] = source_values[cell_index]

    return LinearSystem(matrix, rhs)


def solve_poisson(
    mesh: Mesh,
    source_term: SourceTerm,
    boundary_conditions: Sequence[BoundaryCondition],
    method: str = "gauss_seidel",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> ScalarField:
    """Assemble and solve the 2D Poisson equation on a structured mesh.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D Cartesian mesh.
    source_term : number, callable, array-like, or ScalarField
        The right-hand side ``f`` of ``laplacian(phi) = f``. Pass ``0`` (or a
        zero-valued array) to recover the plain Laplace/diffusion equation.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC (Dirichlet) instances, one per boundary.
    method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``.
    max_iterations : int
        Maximum number of solver iterations.
    tolerance : float
        Iteration stops once the residual between successive iterates
        drops below this value.

    Returns
    -------
    ScalarField
        The solved phi field, attached to ``mesh``.
    """
    system = assemble_poisson_system(mesh, source_term, boundary_conditions)

    if method == "jacobi":
        solver = JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    elif method == "gauss_seidel":
        solver = GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    else:
        raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")

    solution = solver.solve(system)
    return ScalarField(mesh, solution)
