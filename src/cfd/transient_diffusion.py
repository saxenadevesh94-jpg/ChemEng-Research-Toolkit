"""Educational transient (time-dependent) 2D diffusion solver.

This module extends the steady-state diffusion and Poisson solvers from
earlier sprints by adding a time derivative, turning Laplace's equation into
the transient diffusion (heat) equation:

    d(phi)/dt = diffusivity * (d2(phi)/dx2 + d2(phi)/dy2)

on a structured 2D Cartesian mesh with FixedValueBC (Dirichlet) boundaries on
all four sides. Time is integrated implicitly with Backward Euler, which is
unconditionally stable: at every time step the new field phi^(n+1) is found
by solving a linear system built from the *new* time level, rather than by
marching an explicit formula forward.

It reuses the same Mesh, ScalarField, BoundaryCondition, SparseMatrix,
LinearSystem, Equation, and iterative solver infrastructure built in earlier
sprints rather than reimplementing any of it.

Nothing here implements convection, pressure correction, SIMPLE or
Navier-Stokes — those are left for later sprints.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .equation import Equation
from .field import ScalarField
from .linear_system import LinearSystem, SparseMatrix
from .mesh import Mesh
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}


def transient_diffusion_equation() -> Equation:
    """Return an Equation describing the transient diffusion problem this module solves.

    This is purely descriptive bookkeeping — it documents the PDE being
    assembled below using the same symbolic Equation container the rest of
    the CFD package uses, rather than encoding it as a bare string.

    Returns
    -------
    Equation
        ``ddt(phi) = diffusivity * laplacian(phi)``
    """
    equation = Equation(lhs="ddt(phi)", rhs="diffusivity * laplacian(phi)")
    return equation


def _validate_mesh_for_transient_diffusion(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencil used below assumes every interior point has a
    left/right/top/bottom neighbour at a constant spacing, so we verify that
    up front rather than failing confusingly deep inside assembly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("transient_diffusion only supports 2D Cartesian meshes.")

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

    transient_diffusion only supports Dirichlet boundaries, so any other
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
                f"transient_diffusion only supports FixedValueBC (Dirichlet) boundaries, got {type(bc).__name__}."
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


def _validate_initial_field(mesh: Mesh, initial_field: ScalarField) -> None:
    if not isinstance(initial_field, ScalarField):
        raise TypeError(f"initial_field must be a ScalarField, got {type(initial_field).__name__}.")
    if initial_field.mesh is not mesh:
        raise ValueError("initial_field must be attached to the same mesh being solved.")


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _count_time_steps(dt: float, total_time: float) -> int:
    """Work out how many equal steps of size dt fit exactly into total_time."""
    raw_steps = total_time / dt
    n_steps = round(raw_steps)
    if n_steps < 1:
        raise ValueError("total_time must be large enough for at least one time step of size dt.")
    if not np.isclose(raw_steps, n_steps, rtol=1e-6, atol=1e-6):
        raise ValueError("total_time must be an integer multiple of dt.")
    return int(n_steps)


def assemble_transient_diffusion_system(
    mesh: Mesh,
    previous_field: ScalarField,
    boundary_conditions: Sequence[BoundaryCondition],
    diffusivity: float,
    dt: float,
) -> LinearSystem:
    """Assemble the implicit (Backward Euler) linear system for one time step.

    Backward Euler discretises ``d(phi)/dt = diffusivity * laplacian(phi)`` as

        (phi_P^(n+1) - phi_P^n) / dt = diffusivity * (
            (phi_E^(n+1) - 2*phi_P^(n+1) + phi_W^(n+1)) / dx^2
          + (phi_N^(n+1) - 2*phi_P^(n+1) + phi_S^(n+1)) / dy^2
        )

    which rearranges into a linear system for the unknown new-time-level
    values phi^(n+1), with the known old-time-level value phi_P^n on the
    right-hand side.

    Boundary points are handled by direct elimination: their matrix row is
    just ``1 * phi_boundary = fixed_value``, which pins that unknown to the
    FixedValueBC value instead of computing a stencil for it.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh (e.g. from :func:`build_structured_mesh`).
    previous_field : ScalarField
        The solution phi^n at the start of this time step.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances, one per boundary
        ("left", "right", "top", "bottom").
    diffusivity : float
        The (constant, positive) diffusion coefficient.
    dt : float
        The (positive) time step size.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for phi^(n+1).
    """
    _validate_mesh_for_transient_diffusion(mesh)
    _validate_boundary_conditions(boundary_conditions)
    _validate_initial_field(mesh, previous_field)
    diffusivity = _validate_positive_number(diffusivity, "diffusivity")
    dt = _validate_positive_number(dt, "dt")

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

    coeff_x = diffusivity * dt / dx ** 2
    coeff_y = diffusivity * dt / dy ** 2

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

        matrix.set(cell_index, cell_index, 1.0 + 2.0 * (coeff_x + coeff_y))
        matrix.set(cell_index, east, -coeff_x)
        matrix.set(cell_index, west, -coeff_x)
        matrix.set(cell_index, north, -coeff_y)
        matrix.set(cell_index, south, -coeff_y)
        rhs[cell_index] = previous_field.values[cell_index]

    return LinearSystem(matrix, rhs)


def solve_transient_diffusion(
    mesh: Mesh,
    initial_field: ScalarField,
    boundary_conditions: Sequence[BoundaryCondition],
    diffusivity: float,
    dt: float,
    total_time: float,
    method: str = "gauss_seidel",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> ScalarField:
    """March the transient diffusion equation forward from initial_field to total_time.

    At every time step, an implicit (Backward Euler) linear system is
    assembled with :func:`assemble_transient_diffusion_system` and solved
    with the chosen iterative solver, using the previous time step's
    solution as the initial guess (a warm start that speeds up convergence
    since consecutive time steps are usually close to each other).

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh (e.g. from :func:`build_structured_mesh`).
    initial_field : ScalarField
        The field phi at time zero, attached to ``mesh``.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC (Dirichlet) instances, one per boundary.
        These stay fixed for the whole simulation.
    diffusivity : float
        The (constant, positive) diffusion coefficient.
    dt : float
        The (positive) time step size.
    total_time : float
        The (positive) total simulation time. Must be an integer multiple of dt.
    method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``.
    max_iterations : int
        Maximum number of solver iterations per time step.
    tolerance : float
        Iteration stops once the residual between successive iterates
        drops below this value.

    Returns
    -------
    ScalarField
        The solved phi field at ``total_time``, attached to ``mesh``.
    """
    _validate_mesh_for_transient_diffusion(mesh)
    _validate_boundary_conditions(boundary_conditions)
    _validate_initial_field(mesh, initial_field)
    diffusivity = _validate_positive_number(diffusivity, "diffusivity")
    dt = _validate_positive_number(dt, "dt")
    total_time = _validate_positive_number(total_time, "total_time")
    n_steps = _count_time_steps(dt, total_time)

    if method == "jacobi":
        solver = JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    elif method == "gauss_seidel":
        solver = GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    else:
        raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")

    current_field = ScalarField(mesh, initial_field.values.copy())
    for _ in range(n_steps):
        system = assemble_transient_diffusion_system(mesh, current_field, boundary_conditions, diffusivity, dt)
        solution = solver.solve(system, initial_guess=current_field.values)
        current_field = ScalarField(mesh, solution)

    return current_field
