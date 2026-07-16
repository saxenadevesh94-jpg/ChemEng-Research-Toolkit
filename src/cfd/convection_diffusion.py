"""Educational steady-state 2D convection-diffusion solver.

This module extends the steady-state diffusion solver from Sprint 18 by
adding a convective transport term, turning Laplace's equation into the
steady-state convection-diffusion equation:

    u * d(phi)/dx + v * d(phi)/dy = diffusivity * (d2(phi)/dx2 + d2(phi)/dy2)

on a structured 2D Cartesian mesh with FixedValueBC (Dirichlet) boundaries on
all four sides. Convection is discretised with first-order upwind
differencing (the sign of the local velocity component decides which
neighbour is used), while diffusion keeps the central differencing already
used by the diffusion and Poisson solvers.

It reuses the same Mesh, ScalarField, VectorField, BoundaryCondition,
Equation, SparseMatrix, LinearSystem, and iterative solver infrastructure
built in earlier sprints rather than reimplementing any of it.

Nothing here implements pressure correction, SIMPLE or Navier-Stokes —
those are left for later sprints.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .equation import Equation
from .field import ScalarField, VectorField
from .linear_system import LinearSystem, SparseMatrix
from .mesh import Mesh
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}

# Corner cells sit on two boundaries at once (e.g. top-left is on both "left"
# and "top"). Applying BCs in this fixed order — rather than the order the
# caller happened to list them in — makes the outcome deterministic: left/
# right are applied last, so a corner's value always comes from its
# left/right boundary, matching the diffusion solver's convention.
_BOUNDARY_APPLICATION_ORDER = ("top", "bottom", "left", "right")

VelocityLike = Union[
    Tuple[float, float],
    Callable[[float, float], Tuple[float, float]],
    np.ndarray,
    Sequence[float],
    VectorField,
]


def convection_diffusion_equation() -> Equation:
    """Return an Equation describing the convection-diffusion problem this module solves.

    This is purely descriptive bookkeeping — it documents the PDE being
    assembled below using the same symbolic Equation container the rest of
    the CFD package uses, rather than encoding it as a bare string.

    Returns
    -------
    Equation
        ``convection(phi) = diffusivity * laplacian(phi)``
    """
    equation = Equation(lhs="convection(phi)", rhs="diffusivity * laplacian(phi)")
    return equation


def _validate_mesh_for_convection_diffusion(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencil used below assumes every interior point has a
    left/right/top/bottom neighbour at a constant spacing, so we verify that
    up front rather than failing confusingly deep inside assembly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("convection_diffusion only supports 2D Cartesian meshes.")

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

    convection_diffusion only supports Dirichlet boundaries, so any other
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
                f"convection_diffusion only supports FixedValueBC (Dirichlet) boundaries, got "
                f"{type(bc).__name__}."
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


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _evaluate_velocity(mesh: Mesh, velocity: VelocityLike) -> np.ndarray:
    """Turn any supported ``velocity`` representation into a per-cell (u, v) array.

    Parameters
    ----------
    mesh : Mesh
        Mesh the velocity must be evaluated on.
    velocity : (u, v) pair, callable, array-like, or VectorField
        - A ``(u, v)`` pair applies the same velocity everywhere.
        - A callable ``f(x, y)`` returning ``(u, v)`` is evaluated at every
          cell center.
        - An array-like or VectorField supplies one ``(u, v)`` pair per cell
          directly.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(mesh.n_cells, 2)`` with the velocity at each cell.
    """
    if isinstance(velocity, VectorField):
        if velocity.mesh is not mesh:
            raise ValueError("velocity VectorField must be attached to the same mesh being solved.")
        if velocity.values.shape[1] != 2:
            raise ValueError("velocity VectorField must have exactly two components (u, v).")
        return velocity.values.copy()

    if callable(velocity):
        values = np.array(
            [np.asarray(velocity(x, y), dtype=float) for x, y in mesh.cell_centers], dtype=float
        )
        if values.shape != (mesh.n_cells, 2):
            raise ValueError("velocity callable must return a pair (u, v) for every cell.")
        return values

    if isinstance(velocity, (list, tuple, np.ndarray)):
        array = np.asarray(velocity, dtype=float)
        if array.ndim == 1:
            if array.shape != (2,):
                raise ValueError("a constant velocity must be a pair (u, v).")
            return np.tile(array, (mesh.n_cells, 1))
        if array.ndim == 2:
            if array.shape != (mesh.n_cells, 2):
                raise ValueError(f"velocity array must have shape ({mesh.n_cells}, 2).")
            return array
        raise ValueError("velocity array must be 1D (u, v) or 2D (n_cells, 2).")

    raise TypeError(
        "velocity must be a (u, v) pair, a callable f(x, y) -> (u, v), an array-like "
        "of shape (n_cells, 2), or a VectorField."
    )


def assemble_convection_diffusion_system(
    mesh: Mesh,
    velocity: VelocityLike,
    boundary_conditions: Sequence[BoundaryCondition],
    diffusivity: float,
) -> LinearSystem:
    """Assemble the linear system A*phi = b for the steady convection-diffusion equation.

    Interior points combine a first-order upwind discretisation of the
    convective term with the standard five-point central-difference stencil
    for the diffusive term:

        u_P * d(phi)/dx + v_P * d(phi)/dy = diffusivity * (
            (phi[E] - 2*phi[P] + phi[W]) / dx^2
          + (phi[N] - 2*phi[P] + phi[S]) / dy^2
        )

    where the upwind derivative uses the *upstream* neighbour based on the
    sign of the local velocity component:

        d(phi)/dx ~ (phi[P] - phi[W]) / dx   if u_P >= 0
                    (phi[E] - phi[P]) / dx   if u_P < 0

    and likewise for d(phi)/dy with v_P and the N/S neighbours.

    Boundary points are handled by direct elimination: their matrix row is
    just ``1 * phi_boundary = fixed_value``, which pins that unknown to the
    FixedValueBC value instead of computing a stencil for it.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh (e.g. from
        :func:`src.cfd.diffusion_solver.build_structured_mesh`).
    velocity : (u, v) pair, callable, array-like, or VectorField
        The advecting velocity field. See :func:`_evaluate_velocity` for the
        accepted forms.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances, one per boundary
        ("left", "right", "top", "bottom").
    diffusivity : float
        The (constant, positive) diffusion coefficient.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector.
    """
    _validate_mesh_for_convection_diffusion(mesh)
    _validate_boundary_conditions(boundary_conditions)
    diffusivity = _validate_positive_number(diffusivity, "diffusivity")
    velocity_values = _evaluate_velocity(mesh, velocity)

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
    bc_by_boundary = {bc.boundary: bc for bc in boundary_conditions}
    for boundary in _BOUNDARY_APPLICATION_ORDER:
        bc = bc_by_boundary[boundary]
        bc.apply(boundary_field)
        is_boundary[_boundary_cell_indices(mesh, bc.boundary)] = True

    n = mesh.n_cells
    matrix = SparseMatrix(size=n)
    rhs = np.zeros(n)

    diff_coeff_x = diffusivity / dx ** 2
    diff_coeff_y = diffusivity / dy ** 2

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

        u, v = velocity_values[cell_index]

        # Diffusion: central differencing on -diffusivity * laplacian(phi).
        center = 2.0 * (diff_coeff_x + diff_coeff_y)
        east_coeff = -diff_coeff_x
        west_coeff = -diff_coeff_x
        north_coeff = -diff_coeff_y
        south_coeff = -diff_coeff_y

        # Convection: first-order upwind, one axis at a time.
        if u >= 0.0:
            center += u / dx
            west_coeff -= u / dx
        else:
            center -= u / dx
            east_coeff += u / dx

        if v >= 0.0:
            center += v / dy
            south_coeff -= v / dy
        else:
            center -= v / dy
            north_coeff += v / dy

        matrix.set(cell_index, cell_index, center)
        matrix.set(cell_index, east, east_coeff)
        matrix.set(cell_index, west, west_coeff)
        matrix.set(cell_index, north, north_coeff)
        matrix.set(cell_index, south, south_coeff)
        rhs[cell_index] = 0.0

    return LinearSystem(matrix, rhs)


def solve_convection_diffusion(
    mesh: Mesh,
    velocity: VelocityLike,
    boundary_conditions: Sequence[BoundaryCondition],
    diffusivity: float,
    method: str = "gauss_seidel",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> ScalarField:
    """Assemble and solve the steady-state convection-diffusion equation on a mesh.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh (e.g. from
        :func:`src.cfd.diffusion_solver.build_structured_mesh`).
    velocity : (u, v) pair, callable, array-like, or VectorField
        The advecting velocity field. See :func:`_evaluate_velocity` for the
        accepted forms.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC (Dirichlet) instances, one per boundary.
    diffusivity : float
        The (constant, positive) diffusion coefficient.
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
    system = assemble_convection_diffusion_system(mesh, velocity, boundary_conditions, diffusivity)

    if method == "jacobi":
        solver = JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    elif method == "gauss_seidel":
        solver = GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    else:
        raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")

    solution = solver.solve(system)
    return ScalarField(mesh, solution)
