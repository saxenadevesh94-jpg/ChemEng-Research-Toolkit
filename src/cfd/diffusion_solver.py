"""Educational steady-state diffusion (Laplace's equation) solver.

This module ties together everything built in earlier sprints — Mesh,
ScalarField, SparseMatrix, LinearSystem and the boundary condition classes —
into a small, readable solver for the steady-state diffusion equation:

    d2(phi)/dx2 + d2(phi)/dy2 = 0        (i.e. Laplacian(phi) = 0)

on a structured 2D Cartesian mesh with FixedValueBC (Dirichlet) boundaries
on all four sides. This is intentionally the *simplest* PDE problem in CFD:
no time-stepping, no convection, no pressure-velocity coupling. See the
bottom of this file's module docstring in the sprint notes for why that
makes it a good starting point.

Nothing here implements transient diffusion, convection, pressure
correction, SIMPLE or Navier-Stokes — those are left for later sprints.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .field import ScalarField
from .linear_system import LinearSystem, SparseMatrix
from .mesh import Mesh
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}


def build_structured_mesh(nx: int, ny: int, length_x: float = 1.0, length_y: float = 1.0) -> Mesh:
    """Build a simple structured, uniformly-spaced 2D Cartesian mesh.

    The mesh is a regular grid of ``nx`` by ``ny`` points spanning
    ``[0, length_x]`` by ``[0, length_y]``. Points are stored as mesh
    "cells" (matching how the rest of the CFD package already treats
    Mesh.cell_centers as a simple point cloud rather than true finite-volume
    cells), and faces are created between every pair of horizontally or
    vertically adjacent points so that the Mesh class's face bookkeeping
    stays consistent.

    Parameters
    ----------
    nx, ny : int
        Number of grid points along x and y. Each must be at least 3 so a
        finite-difference stencil has room to work with (one point on each
        side of an interior point).
    length_x, length_y : float
        Physical size of the domain along x and y.

    Returns
    -------
    Mesh
        A structured mesh ready to use with :func:`assemble_diffusion_system`.
    """
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise TypeError("nx and ny must be integers.")
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must each be at least 3 to support a finite-difference stencil.")
    if length_x <= 0 or length_y <= 0:
        raise ValueError("length_x and length_y must be positive.")

    x = np.linspace(0.0, float(length_x), nx)
    y = np.linspace(0.0, float(length_y), ny)

    # Row-major layout: point (i, j) lives at flat index j * nx + i.
    cell_centers = np.array([[x[i], y[j]] for j in range(ny) for i in range(nx)])

    def flat_index(i: int, j: int) -> int:
        return j * nx + i

    face_centers = []
    owner_cells = []
    neighbour_cells = []

    for j in range(ny):
        for i in range(nx):
            if i < nx - 1:
                owner_cells.append(flat_index(i, j))
                neighbour_cells.append(flat_index(i + 1, j))
                face_centers.append([(x[i] + x[i + 1]) / 2.0, y[j]])
            if j < ny - 1:
                owner_cells.append(flat_index(i, j))
                neighbour_cells.append(flat_index(i, j + 1))
                face_centers.append([x[i], (y[j] + y[j + 1]) / 2.0])

    return Mesh(
        cell_centers=cell_centers,
        face_centers=np.array(face_centers),
        face_areas=np.ones(len(face_centers)),
        cell_volumes=np.ones(nx * ny),
        owner_cells=np.array(owner_cells, dtype=int),
        neighbour_cells=np.array(neighbour_cells, dtype=int),
    )


def _validate_mesh_for_diffusion(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencil used below assumes every interior point has a
    left/right/top/bottom neighbour at a constant spacing, so we verify that
    up front rather than failing confusingly deep inside assembly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("diffusion_solver only supports 2D Cartesian meshes.")

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

    This sprint only supports FixedValueBC (Dirichlet) boundaries, so any
    other BoundaryCondition subclass — such as ZeroGradientBC — is rejected
    here rather than producing an incorrectly assembled system.
    """
    if not isinstance(boundary_conditions, (list, tuple)):
        raise TypeError("boundary_conditions must be a list or tuple of FixedValueBC instances.")
    if len(boundary_conditions) != 4:
        raise ValueError("exactly four boundary conditions are required: left, right, top, bottom.")

    seen = set()
    for bc in boundary_conditions:
        if not isinstance(bc, FixedValueBC):
            raise TypeError(
                f"diffusion_solver only supports FixedValueBC boundaries, got {type(bc).__name__}."
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


def assemble_diffusion_system(mesh: Mesh, boundary_conditions: Sequence[BoundaryCondition]) -> LinearSystem:
    """Assemble the linear system A*phi = b for Laplacian(phi) = 0.

    Interior points get the standard five-point finite-difference stencil
    for the 2D Laplacian:

        (phi[E] - 2*phi[P] + phi[W]) / dx^2
      + (phi[N] - 2*phi[P] + phi[S]) / dy^2 = 0

    Boundary points are handled by direct elimination: their matrix row is
    just ``1 * phi_boundary = fixed_value``, which pins that unknown to the
    FixedValueBC value instead of computing a stencil for it.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh (e.g. from :func:`build_structured_mesh`).
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances, one per boundary
        ("left", "right", "top", "bottom").

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector.
    """
    _validate_mesh_for_diffusion(mesh)
    _validate_boundary_conditions(boundary_conditions)

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
        rhs[cell_index] = 0.0

    return LinearSystem(matrix, rhs)


def solve_diffusion(
    mesh: Mesh,
    boundary_conditions: Sequence[BoundaryCondition],
    method: str = "jacobi",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> ScalarField:
    """Assemble and solve the steady-state diffusion equation on a mesh.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh (e.g. from :func:`build_structured_mesh`).
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances, one per boundary.
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
    system = assemble_diffusion_system(mesh, boundary_conditions)

    if method == "jacobi":
        solver = JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    elif method == "gauss_seidel":
        solver = GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    else:
        raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")

    solution = solver.solve(system)
    return ScalarField(mesh, solution)
