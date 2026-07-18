"""Educational steady, incompressible, laminar 2D Navier-Stokes solver.

This module is the user-facing entry point for the CFD package. Earlier
sprints already assembled everything a steady incompressible Navier-Stokes
solve needs:

    * :mod:`~src.cfd.mesh`, :mod:`~src.cfd.field` and
      :mod:`~src.cfd.boundary_conditions` for the mesh and field containers,
    * :mod:`~src.cfd.convection_diffusion` for the momentum prediction step,
    * :mod:`~src.cfd.pressure_correction` for the pressure-velocity coupling,
    * :mod:`~src.cfd.simple_solver` for the outer SIMPLE iteration that ties
      momentum prediction and pressure correction together.

``simple_solver.solve_simple`` already solves the governing equations

    u * d(u)/dx + v * d(u)/dy = -d(p)/dx + viscosity * laplacian(u)
    u * d(v)/dx + v * d(v)/dy = -d(p)/dy + viscosity * laplacian(v)
    d(u)/dx + d(v)/dy = 0                                              (continuity)

in *kinematic* form: its ``viscosity`` is a kinematic viscosity and its
``pressure`` is really pressure divided by density (so the momentum equation
does not need a density term). This module wraps that solver with the
dimensional quantities engineers actually work with — density and dynamic
viscosity — and with the ability to anchor the resulting pressure field to a
chosen reference cell and value, since an incompressible pressure field is
only ever defined up to an additive constant.

No new PDE discretisation, matrix assembly or iterative solving happens
here: this module only converts units on the way in and out of
:func:`~src.cfd.simple_solver.solve_simple` and performs the additive
pressure shift.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .boundary_conditions import BoundaryCondition
from .field import ScalarField, VectorField
from .mesh import Mesh
from .simple_solver import solve_simple


class NavierStokesResult:
    """Container for the outcome of a steady incompressible Navier-Stokes solve.

    Parameters
    ----------
    velocity : VectorField
        The final velocity field.
    pressure : ScalarField
        The final, dimensional pressure field (already anchored to the
        requested pressure reference point).
    density : float
        The (constant, positive) fluid density used for the solve.
    dynamic_viscosity : float
        The (constant, positive) dynamic viscosity used for the solve.
    iterations_run : int
        Number of outer SIMPLE iterations actually run.
    converged : bool
        Whether every SIMPLE residual dropped below the requested tolerance.
    residual_history : list of dict
        One residual dictionary per outer iteration, as returned by
        :func:`~src.cfd.simple_solver.compute_residuals`.
    """

    def __init__(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        density: float,
        dynamic_viscosity: float,
        iterations_run: int,
        converged: bool,
        residual_history: List[Dict[str, float]],
    ) -> None:
        self.velocity = velocity
        self.pressure = pressure
        self.density = density
        self.dynamic_viscosity = dynamic_viscosity
        self.iterations_run = iterations_run
        self.converged = converged
        self.residual_history = residual_history

    def __repr__(self) -> str:
        final_residual = self.residual_history[-1] if self.residual_history else None
        return (
            f"NavierStokesResult(iterations_run={self.iterations_run}, "
            f"converged={self.converged}, final_residual={final_residual})"
        )


def _validate_mesh_type(mesh: Mesh) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _validate_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    return float(value)


def _validate_reference_cell(reference_cell: object, n_cells: int) -> int:
    if isinstance(reference_cell, bool) or not isinstance(reference_cell, int):
        raise TypeError(f"pressure_reference_cell must be an integer, got {type(reference_cell).__name__}.")
    if not (0 <= reference_cell < n_cells):
        raise ValueError(
            f"pressure_reference_cell {reference_cell} is out of range for a mesh with {n_cells} cells."
        )
    return reference_cell


def _validate_initial_pressure(mesh: Mesh, initial_pressure: Optional[ScalarField]) -> Optional[ScalarField]:
    if initial_pressure is None:
        return None
    if not isinstance(initial_pressure, ScalarField):
        raise TypeError(f"initial_pressure must be a ScalarField, got {type(initial_pressure).__name__}.")
    if initial_pressure.mesh is not mesh:
        raise ValueError("initial_pressure must be attached to the same mesh being solved.")
    return initial_pressure


def _anchor_pressure(
    pressure: ScalarField,
    density: float,
    reference_cell: int,
    reference_value: float,
) -> ScalarField:
    """Shift a kinematic pressure field to a dimensional field anchored at one cell.

    Incompressible pressure is only defined up to an additive constant (see
    :mod:`~src.cfd.pressure_correction`'s module docstring), so any constant
    shift of ``pressure`` solves the same momentum and continuity equations.
    This picks the shift that makes the dimensional pressure equal
    ``reference_value`` at ``reference_cell``, then converts the rest of the
    field from kinematic (p / density) to dimensional (p) units using that
    same shift.
    """
    shift = reference_value / density - pressure.values[reference_cell]
    dimensional_values = density * (pressure.values + shift)
    return ScalarField(pressure.mesh, dimensional_values)


def solve_navier_stokes(
    mesh: Mesh,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    density: float,
    dynamic_viscosity: float,
    pressure_reference_cell: int = 0,
    pressure_reference_value: float = 0.0,
    initial_velocity: Optional[VectorField] = None,
    initial_pressure: Optional[ScalarField] = None,
    velocity_relaxation: float = 0.7,
    pressure_relaxation: float = 0.3,
    max_outer_iterations: int = 200,
    outer_tolerance: float = 1e-6,
    linear_method: str = "gauss_seidel",
    linear_max_iterations: int = 2000,
    linear_tolerance: float = 1e-8,
) -> NavierStokesResult:
    """Solve the steady, incompressible, laminar 2D Navier-Stokes equations.

    This is a thin dimensional wrapper around
    :func:`~src.cfd.simple_solver.solve_simple`: it converts ``density`` and
    ``dynamic_viscosity`` into the kinematic viscosity SIMPLE expects,
    forwards everything else unchanged, and converts the resulting kinematic
    pressure field back into a dimensional pressure field anchored at
    ``pressure_reference_cell``. No momentum, pressure-correction or outer
    SIMPLE-loop logic is reimplemented here.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniformly-spaced 2D Cartesian mesh (e.g. from
        :func:`~src.cfd.diffusion_solver.build_structured_mesh`).
    u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC (Dirichlet) instances each, one per
        boundary ("left", "right", "top", "bottom").
    density : float
        The (constant, positive) fluid density.
    dynamic_viscosity : float
        The (constant, positive) dynamic viscosity.
    pressure_reference_cell : int
        Index of the mesh cell whose pressure is pinned to
        ``pressure_reference_value``. Defaults to ``0``.
    pressure_reference_value : float
        The dimensional pressure value assigned to
        ``pressure_reference_cell``. Defaults to ``0.0``.
    initial_velocity : VectorField, optional
        Starting velocity guess. Defaults to zero everywhere.
    initial_pressure : ScalarField, optional
        Starting dimensional pressure guess. Defaults to zero everywhere.
    velocity_relaxation : float
        Under-relaxation factor applied to the momentum prediction, in
        (0, 1]. Defaults to ``0.7``.
    pressure_relaxation : float
        Under-relaxation factor applied to the pressure update, in (0, 1].
        Defaults to ``0.3``.
    max_outer_iterations : int
        Maximum number of SIMPLE outer iterations.
    outer_tolerance : float
        Outer iteration stops once every SIMPLE residual drops below this
        value.
    linear_method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``, used for every inner
        linear solve (momentum prediction and pressure correction).
    linear_max_iterations : int
        Maximum number of iterations for each inner linear solve.
    linear_tolerance : float
        Convergence tolerance for each inner linear solve.

    Returns
    -------
    NavierStokesResult
        The final velocity and dimensional pressure fields, plus convergence
        bookkeeping.
    """
    _validate_mesh_type(mesh)
    density = _validate_positive_number(density, "density")
    dynamic_viscosity = _validate_positive_number(dynamic_viscosity, "dynamic_viscosity")
    pressure_reference_cell = _validate_reference_cell(pressure_reference_cell, mesh.n_cells)
    pressure_reference_value = _validate_number(pressure_reference_value, "pressure_reference_value")
    initial_pressure = _validate_initial_pressure(mesh, initial_pressure)

    kinematic_viscosity = dynamic_viscosity / density
    kinematic_initial_pressure = (
        None if initial_pressure is None else ScalarField(mesh, initial_pressure.values / density)
    )

    result = solve_simple(
        mesh,
        u_boundary_conditions,
        v_boundary_conditions,
        kinematic_viscosity,
        initial_velocity=initial_velocity,
        initial_pressure=kinematic_initial_pressure,
        velocity_relaxation=velocity_relaxation,
        pressure_relaxation=pressure_relaxation,
        max_outer_iterations=max_outer_iterations,
        outer_tolerance=outer_tolerance,
        linear_method=linear_method,
        linear_max_iterations=linear_max_iterations,
        linear_tolerance=linear_tolerance,
    )

    dimensional_pressure = _anchor_pressure(
        result.pressure, density, pressure_reference_cell, pressure_reference_value
    )

    return NavierStokesResult(
        velocity=result.velocity,
        pressure=dimensional_pressure,
        density=density,
        dynamic_viscosity=dynamic_viscosity,
        iterations_run=result.iterations_run,
        converged=result.converged,
        residual_history=result.residual_history,
    )
