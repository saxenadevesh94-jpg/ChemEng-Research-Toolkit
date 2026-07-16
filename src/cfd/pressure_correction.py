"""Educational 2D pressure correction equation solver (SIMPLE-style).

This module builds the pressure correction equation used by pressure-
velocity coupling algorithms such as SIMPLE and by projection methods more
generally:

    laplacian(p_prime) = divergence(velocity_star) / dt

on a structured 2D Cartesian mesh with homogeneous Neumann (zero-gradient)
pressure correction boundaries on all four sides. ``velocity_star`` is the
intermediate (generally non-divergence-free) velocity field produced by a
momentum predictor step; solving this equation for ``p_prime`` gives the
correction needed to project ``velocity_star`` onto a divergence-free field.

A pure-Neumann Laplacian problem is only defined up to an additive constant
(no boundary anchors ``p_prime`` to any particular value), so the linear
system assembled here also pins a single reference cell's correction to
zero. This is the same, standard trick used to make pressure-correction
systems solvable in real CFD codes, and it is the smallest possible fix:
it removes the singularity without changing the physically meaningful
pressure *gradients*.

It reuses the same Mesh, ScalarField, VectorField, BoundaryCondition,
Equation, SparseMatrix, LinearSystem, the divergence operator, and the
iterative solver infrastructure built in earlier sprints rather than
reimplementing any of it.

Nothing here implements the full SIMPLE loop, velocity correction, momentum
equations or Navier-Stokes — those are left for later sprints.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .boundary_conditions import BoundaryCondition, ZeroGradientBC
from .equation import Equation
from .field import ScalarField, VectorField
from .linear_system import LinearSystem, SparseMatrix
from .mesh import Mesh
from .operators import divergence
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}

# Corner cells sit on two boundaries at once (e.g. top-left is on both "left"
# and "top"). Applying BCs in this fixed order — rather than the order the
# caller happened to list them in — makes the outcome deterministic: left/
# right are applied last, so a corner's mirror direction always comes from
# its left/right boundary, matching the convention used by the other CFD
# solvers in this package.
_BOUNDARY_APPLICATION_ORDER = ("top", "bottom", "left", "right")


def pressure_correction_equation() -> Equation:
    """Return an Equation describing the pressure correction problem this module solves.

    This is purely descriptive bookkeeping — it documents the PDE being
    assembled below using the same symbolic Equation container the rest of
    the CFD package uses, rather than encoding it as a bare string.

    Returns
    -------
    Equation
        ``laplacian(p_prime) = divergence(velocity_star) / dt``
    """
    equation = Equation(lhs="laplacian(p_prime)", rhs="divergence(velocity_star) / dt")
    return equation


def _validate_mesh_for_pressure_correction(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencil used below assumes every interior point has a
    left/right/top/bottom neighbour at a constant spacing, so we verify that
    up front rather than failing confusingly deep inside assembly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("pressure_correction only supports 2D Cartesian meshes.")

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
    """Check that exactly one ZeroGradientBC is given for each of the four sides.

    pressure_correction only supports homogeneous Neumann (zero-gradient)
    boundaries, so any other BoundaryCondition subclass — such as
    FixedValueBC — is rejected here rather than producing an incorrectly
    assembled system.
    """
    if not isinstance(boundary_conditions, (list, tuple)):
        raise TypeError("boundary_conditions must be a list or tuple of ZeroGradientBC instances.")
    if len(boundary_conditions) != 4:
        raise ValueError("exactly four boundary conditions are required: left, right, top, bottom.")

    seen = set()
    for bc in boundary_conditions:
        if not isinstance(bc, ZeroGradientBC):
            raise TypeError(
                f"pressure_correction only supports ZeroGradientBC (homogeneous Neumann) boundaries, "
                f"got {type(bc).__name__}."
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


def _validate_velocity_star(mesh: Mesh, velocity_star: VectorField) -> None:
    if not isinstance(velocity_star, VectorField):
        raise TypeError(f"velocity_star must be a VectorField, got {type(velocity_star).__name__}.")
    if velocity_star.mesh is not mesh:
        raise ValueError("velocity_star must be attached to the same mesh being solved.")
    if velocity_star.values.shape[1] != 2:
        raise ValueError("velocity_star must have exactly two components (u, v).")


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


# Sentinel for "the caller did not pass reference_cell". A mesh corner (the
# static default this used to be) is never referenced by any other row in
# the assembled system — it is never an interior cell's N/E/S/W neighbour,
# nor any boundary cell's zero-gradient mirror target — so pinning one
# leaves the matrix singular instead of removing its null space. The safe
# default therefore has to be resolved per-mesh (see assemble_pressure_correction),
# not hard-coded, hence a sentinel rather than a fixed integer default.
_UNSET_REFERENCE_CELL = object()


def _validate_reference_cell(reference_cell: object, n_cells: int) -> object:
    if reference_cell is _UNSET_REFERENCE_CELL:
        return reference_cell
    if isinstance(reference_cell, bool) or not isinstance(reference_cell, int):
        raise TypeError(f"reference_cell must be an integer, got {type(reference_cell).__name__}.")
    if not (0 <= reference_cell < n_cells):
        raise ValueError(f"reference_cell {reference_cell} is out of range for a mesh with {n_cells} cells.")
    return reference_cell


def assemble_pressure_correction(
    mesh: Mesh,
    velocity_star: VectorField,
    boundary_conditions: Sequence[BoundaryCondition],
    dt: float = 1.0,
    reference_cell: int = _UNSET_REFERENCE_CELL,
) -> LinearSystem:
    """Assemble the linear system A*p_prime = b for laplacian(p_prime) = divergence(velocity_star) / dt.

    Interior points get the standard five-point finite-difference stencil for
    the 2D Laplacian, with the (reused) divergence operator's output at that
    point on the right-hand side:

        (p'[E] - 2*p'[P] + p'[W]) / dx^2
      + (p'[N] - 2*p'[P] + p'[S]) / dy^2 = divergence(velocity_star)[P] / dt

    Boundary points get a homogeneous-Neumann (zero-gradient) row instead of
    a stencil: ``p'_boundary - p'_neighbour = 0``, pinning the boundary value
    equal to its nearest interior neighbour so no gradient is created across
    the boundary. This mirrors how :class:`ZeroGradientBC` already treats an
    existing field, expressed here as a matrix row instead.

    Because that leaves the system singular (any constant p_prime field
    solves it), ``reference_cell`` is pinned directly to zero, which is the
    standard way of making a pure-Neumann pressure equation solvable.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    velocity_star : VectorField
        The intermediate velocity field whose divergence drives the pressure
        correction. Must be attached to ``mesh`` and have exactly two
        components (u, v).
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four ZeroGradientBC instances, one per boundary
        ("left", "right", "top", "bottom").
    dt : float
        The (positive) time step (or relaxation) scale relating the pressure
        correction to the velocity divergence. Defaults to ``1.0``.
    reference_cell : int
        Index of the cell whose pressure correction is pinned to zero to
        remove the null space of the pure-Neumann system. Must not be a mesh
        corner (see note above). Defaults to an interior cell chosen
        automatically for the given mesh.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector.
    """
    _validate_mesh_for_pressure_correction(mesh)
    _validate_boundary_conditions(boundary_conditions)
    _validate_velocity_star(mesh, velocity_star)
    dt = _validate_positive_number(dt, "dt")
    reference_cell = _validate_reference_cell(reference_cell, mesh.n_cells)

    source_values = divergence(velocity_star).values / dt

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

    if reference_cell is _UNSET_REFERENCE_CELL:
        # The interior cell at grid position (1, 1) always exists (every
        # axis has at least three points, per _validate_mesh_for_pressure_correction)
        # and is always referenced by its neighbours' stencils, so it is
        # always a safe pin — unlike a corner cell (see _UNSET_REFERENCE_CELL).
        reference_cell = int(grid_index[1, 1])

    # Work out which cells sit on the boundary and which neighbour each one
    # should mirror (zero-gradient), using the same fixed corner precedence
    # as the rest of the package.
    is_boundary = np.zeros(mesh.n_cells, dtype=bool)
    mirror_neighbor = np.full(mesh.n_cells, -1, dtype=int)
    for boundary in _BOUNDARY_APPLICATION_ORDER:
        for cell_index in _boundary_cell_indices(mesh, boundary):
            is_boundary[cell_index] = True
            ix, iy = ix_of[cell_index], iy_of[cell_index]
            if boundary == "left":
                neighbor_ix, neighbor_iy = ix + 1, iy
            elif boundary == "right":
                neighbor_ix, neighbor_iy = ix - 1, iy
            elif boundary == "bottom":
                neighbor_ix, neighbor_iy = ix, iy + 1
            else:  # "top"
                neighbor_ix, neighbor_iy = ix, iy - 1
            mirror_neighbor[cell_index] = grid_index[neighbor_ix, neighbor_iy]

    n = mesh.n_cells
    matrix = SparseMatrix(size=n)
    rhs = np.zeros(n)

    coeff_x = 1.0 / dx ** 2
    coeff_y = 1.0 / dy ** 2

    for cell_index in range(n):
        if cell_index == reference_cell:
            # Reference elimination: pin one cell to remove the null space
            # of the pure-Neumann system.
            matrix.set(cell_index, cell_index, 1.0)
            rhs[cell_index] = 0.0
            continue

        if is_boundary[cell_index]:
            # Zero-gradient elimination: p_prime at this cell mirrors its
            # nearest interior neighbour, creating no gradient at the wall.
            matrix.set(cell_index, cell_index, 1.0)
            matrix.set(cell_index, int(mirror_neighbor[cell_index]), -1.0)
            rhs[cell_index] = 0.0
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


def solve_pressure_correction(
    mesh: Mesh,
    velocity_star: VectorField,
    boundary_conditions: Sequence[BoundaryCondition],
    dt: float = 1.0,
    reference_cell: int = _UNSET_REFERENCE_CELL,
    method: str = "gauss_seidel",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> ScalarField:
    """Assemble and solve the pressure correction equation on a structured mesh.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D Cartesian mesh.
    velocity_star : VectorField
        The intermediate velocity field whose divergence drives the pressure
        correction. Must be attached to ``mesh`` and have exactly two
        components (u, v).
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four ZeroGradientBC (homogeneous Neumann) instances, one per
        boundary.
    dt : float
        The (positive) time step (or relaxation) scale relating the pressure
        correction to the velocity divergence. Defaults to ``1.0``.
    reference_cell : int
        Index of the cell whose pressure correction is pinned to zero.
        Defaults to an interior cell chosen automatically for the given mesh.
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
        The solved p_prime field, attached to ``mesh``.
    """
    system = assemble_pressure_correction(mesh, velocity_star, boundary_conditions, dt, reference_cell)

    if method == "jacobi":
        solver = JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    elif method == "gauss_seidel":
        solver = GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    else:
        raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")

    solution = solver.solve(system)
    return ScalarField(mesh, solution)
