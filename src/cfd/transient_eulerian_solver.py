"""Educational transient (first-order implicit) Eulerian two-fluid solver.

Sprint 32 adds time-stepping on top of the steady two-fluid (Euler-Euler)
solver built in :mod:`~src.cfd.eulerian_solver`. Each phase's momentum
equation

    alpha_k * rho_k * (U_k . grad) U_k =
        -alpha_k * grad(p) + alpha_k * mu_k * laplacian(U_k)
        + K * (U_other - U_k)

gains a time-derivative term, ``d(U_k)/dt``, discretised with first-order
implicit (Backward Euler) time integration - the same scheme already used by
:mod:`~src.cfd.transient_diffusion` for the transient diffusion equation:

    (U_k^(n+1) - U_k^n) / dt + (U_k^(n+1) . grad) U_k^(n+1) =
        -grad(p^(n+1)) / rho_k + (mu_k / rho_k) * laplacian(U_k^(n+1))
        + K * (U_other^(n+1) - U_k^(n+1)) / (alpha_k * rho_k)

At every time step this module reuses
:func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system` to assemble
the steady convection/diffusion/pressure/drag terms, then adds ``1/dt`` to
every interior row's matrix diagonal and ``U_k^n / dt`` to its right-hand
side - exactly the pattern
:func:`~src.cfd.transient_diffusion.assemble_transient_diffusion_system`
already uses for the transient diffusion equation. The resulting linear
system is solved with the same outer (Gauss-Seidel phase-coupling,
under-relaxed) iteration :func:`~src.cfd.eulerian_solver.solve_eulerian_two_fluid`
already uses, once per time step, so that the convective/drag nonlinearity
is still resolved implicitly within each step.

The shared pressure field is re-solved once per time step from the
volume-fraction-weighted mixture momentum equation
(:meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.solve_mixture_flow`,
itself the existing SIMPLE/Navier-Stokes solver), warm-started from the
previous step's mixture velocity and pressure. This is a quasi-steady
treatment of pressure - the mixture SIMPLE solve itself has no time
derivative - which keeps this module a thin, honest extension of the
existing steady infrastructure rather than a reimplementation of a fully
transient pressure-velocity coupling.

Volume-fraction transport is, as already noted in
:mod:`~src.cfd.multiphase`, left for a later sprint: each phase's volume
fraction is held fixed for the whole simulation. It is still recorded at
every time step (as required of any stored history) so callers get a
complete, consistently-shaped time history even though the values do not
yet change.

An optional :class:`~src.cfd.packed_bed.PackedBedProperties` may be
supplied to add Ergun-equation packed-bed resistance to both phases'
momentum equations at every time step, reusing
:func:`~src.cfd.packed_bed.assemble_packed_bed_phase_momentum_system`
instead of reimplementing that resistance term. The
:class:`TransientEulerianTwoFluidResult` can also build a
:class:`~src.cfd.force_accounting.ForceAccounting` report for any stored
time step, reusing that module's bookkeeping rather than reimplementing it.

Boundary conditions may be given as a fixed (steady) list of four
FixedValueBC instances, used unchanged for the whole simulation, or as a
callable ``time -> list of four FixedValueBC`` that is re-evaluated at every
time step, allowing transient boundary conditions such as a ramped inlet
velocity.

This module reuses the same Mesh, ScalarField, VectorField,
BoundaryCondition, Equation, LinearSystem, EulerianMultiphaseSystem, Phase,
Navier-Stokes/SIMPLE solver, two-fluid drag model, packed-bed resistance
model and force accounting infrastructure built in earlier sprints rather
than reimplementing any of it.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .equation import Equation
from .eulerian_solver import (
    assemble_phase_momentum_system,
    interphase_drag_coefficient,
    relative_velocity_magnitude,
)
from .field import ScalarField, VectorField
from .force_accounting import ForceAccounting
from .linear_system import LinearSystem
from .mesh import Mesh
from .multiphase import EulerianMultiphaseSystem, Phase
from .operators import gradient
from .packed_bed import ErgunResistanceModel, PackedBedProperties, assemble_packed_bed_phase_momentum_system
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}
_COMPONENT_INDEX = {"u": 0, "v": 1}

# A schedule that is a plain list/tuple of BoundaryCondition instances is
# "steady": the same boundary conditions apply at every time step. A
# schedule that is a callable ``time -> list of BoundaryCondition`` is
# "transient": it is re-evaluated (and re-validated) at every time step.
BoundaryConditionSchedule = Union[Sequence[BoundaryCondition], Callable[[float], Sequence[BoundaryCondition]]]


# ---------------------------------------------------------------------------
# Descriptive bookkeeping
# ---------------------------------------------------------------------------

def transient_phase_momentum_equation(phase_name: str, other_phase_name: str) -> Equation:
    """Return an Equation describing one phase's transient momentum equation.

    This is purely descriptive bookkeeping - it documents the PDE
    :func:`solve_transient_eulerian_two_fluid` actually assembles and
    solves at every time step, using the same symbolic Equation container
    the rest of the CFD package uses.

    Parameters
    ----------
    phase_name : str
        Name of the phase this equation describes.
    other_phase_name : str
        Name of the other phase it exchanges momentum with through drag.

    Returns
    -------
    Equation
        ``ddt(U_<phase_name>) + convection(U_<phase_name>) =
        -gradient(p) / rho_<phase_name> +
        viscosity_<phase_name> * laplacian(U_<phase_name>) +
        drag(U_<other_phase_name> - U_<phase_name>) / (alpha_<phase_name> * rho_<phase_name>)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    if not isinstance(other_phase_name, str) or not other_phase_name.strip():
        raise ValueError("other_phase_name must be a non-empty string.")
    return Equation(
        lhs=f"ddt(U_{phase_name}) + convection(U_{phase_name})",
        rhs=(
            f"-gradient(p) / rho_{phase_name} "
            f"+ viscosity_{phase_name} * laplacian(U_{phase_name}) "
            f"+ drag(U_{other_phase_name} - U_{phase_name}) / (alpha_{phase_name} * rho_{phase_name})"
        ),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_mesh_type(mesh: Mesh) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be an instance of Mesh, got {type(mesh).__name__}.")


def _validate_structured_mesh(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencils used by momentum assembly assume every interior
    point has a left/right/top/bottom neighbour at a constant spacing, so we
    verify that up front rather than failing confusingly deep inside
    assembly.
    """
    _validate_mesh_type(mesh)
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("solve_transient_eulerian_two_fluid only supports 2D Cartesian meshes.")

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


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _validate_relaxation_factor(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not (0.0 < number <= 1.0):
        raise ValueError(f"{name} must be within the range (0, 1].")
    return number


def _validate_positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _validate_component(component: str) -> int:
    if component not in _COMPONENT_INDEX:
        raise ValueError("component must be either 'u' or 'v'.")
    return _COMPONENT_INDEX[component]


def _validate_velocity_field(field: object, name: str) -> None:
    if not isinstance(field, VectorField):
        raise TypeError(f"{name} must be a VectorField, got {type(field).__name__}.")
    if field.values.shape[1] != 2:
        raise ValueError(f"{name} must have exactly two components (u, v).")


def _validate_velocity_boundary_conditions(boundary_conditions: Sequence[BoundaryCondition], name: str) -> None:
    """Check that exactly one FixedValueBC is given for each of the four sides.

    Momentum assembly only supports Dirichlet velocity boundaries, so any
    other BoundaryCondition subclass - such as ZeroGradientBC - is rejected
    here rather than producing an incorrectly assembled system.
    """
    if not isinstance(boundary_conditions, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of FixedValueBC instances.")
    if len(boundary_conditions) != 4:
        raise ValueError(f"{name} must contain exactly four boundary conditions: left, right, top, bottom.")

    seen = set()
    for bc in boundary_conditions:
        if not isinstance(bc, FixedValueBC):
            raise TypeError(
                f"{name} only supports FixedValueBC (Dirichlet) boundaries, got {type(bc).__name__}."
            )
        if bc.boundary in seen:
            raise ValueError(f"boundary '{bc.boundary}' was specified more than once in {name}.")
        seen.add(bc.boundary)

    missing = _REQUIRED_BOUNDARIES - seen
    if missing:
        raise ValueError(f"{name} is missing boundary condition(s) for: {', '.join(sorted(missing))}")


def _validate_boundary_condition_schedule(schedule: object, name: str) -> None:
    """Validate a boundary condition schedule: either a fixed list, or a callable.

    A callable schedule's returned boundary conditions can only be checked
    once it is actually called with a specific time, so that check is
    deferred to :func:`_resolve_boundary_conditions`.
    """
    if callable(schedule):
        return
    _validate_velocity_boundary_conditions(schedule, name)


def _resolve_boundary_conditions(
    schedule: BoundaryConditionSchedule, time: float, name: str
) -> List[BoundaryCondition]:
    """Return the concrete boundary conditions to use at the given time."""
    if callable(schedule):
        resolved = schedule(time)
        _validate_velocity_boundary_conditions(resolved, name)
        return list(resolved)
    return list(schedule)


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


def _interior_mask(mesh: Mesh, boundary_conditions: Sequence[BoundaryCondition]) -> np.ndarray:
    """Return a boolean mask that is False on every cell touched by a boundary."""
    mask = np.ones(mesh.n_cells, dtype=bool)
    for bc in boundary_conditions:
        mask[_boundary_cell_indices(mesh, bc.boundary)] = False
    return mask


def _build_solver(method: str, max_iterations: int, tolerance: float):
    if method == "jacobi":
        return JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    if method == "gauss_seidel":
        return GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")


def _count_time_steps(dt: float, total_time: float) -> int:
    """Work out how many equal steps of size dt fit exactly into total_time."""
    raw_steps = total_time / dt
    n_steps = round(raw_steps)
    if n_steps < 1:
        raise ValueError("total_time must be large enough for at least one time step of size dt.")
    if not np.isclose(raw_steps, n_steps, rtol=1e-6, atol=1e-6):
        raise ValueError("total_time must be an integer multiple of dt.")
    return int(n_steps)


# ---------------------------------------------------------------------------
# Transient per-phase momentum assembly
# ---------------------------------------------------------------------------

def assemble_transient_phase_momentum_system(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    previous_velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    boundary_conditions: Sequence[BoundaryCondition],
    component: str,
    dt: float,
    bed_properties: Optional[PackedBedProperties] = None,
) -> LinearSystem:
    """Assemble one time step's implicit momentum system for a two-fluid phase.

    Adds a first-order implicit (Backward Euler) time derivative on top of
    the existing steady two-fluid phase momentum assembly
    (:func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`, or,
    when ``bed_properties`` is given,
    :func:`~src.cfd.packed_bed.assemble_packed_bed_phase_momentum_system`),
    exactly as :func:`~src.cfd.transient_diffusion.assemble_transient_diffusion_system`
    adds one to the steady diffusion assembly:

        (U_P^(n+1) - U_P^n) / dt + <steady momentum system> = 0

    which rearranges into a linear system with ``1/dt`` added to the
    diagonal and ``U_P^n / dt`` added to the right-hand side, at every
    interior row. Boundary rows are left untouched (they already pin
    velocity to its FixedValueBC value).

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    phase : Phase
        The phase this momentum equation is being assembled for.
    velocity : VectorField
        The current (within-time-step) velocity guess for ``phase``, used to
        evaluate the convective term.
    previous_velocity : VectorField
        The velocity of ``phase`` at the start of this time step, U^n, used
        by the implicit time-derivative term.
    other_phase_velocity : VectorField
        The current velocity guess for the other phase, used by the
        (implicit) drag source term.
    drag_coefficient : ScalarField
        The interphase momentum exchange coefficient, K.
    pressure : ScalarField
        The shared pressure field at the new time level.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances for this velocity component.
    component : str
        Either ``"u"`` or ``"v"``.
    dt : float
        The (positive) time step size.
    bed_properties : PackedBedProperties, optional
        If given, packed-bed (Ergun-equation) resistance is added on top of
        the usual convection/diffusion/pressure/drag terms.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for this component.
    """
    index = _validate_component(component)
    _validate_structured_mesh(mesh)
    _validate_velocity_field(previous_velocity, "previous_velocity")
    dt = _validate_positive_number(dt, "dt")
    if bed_properties is not None and not isinstance(bed_properties, PackedBedProperties):
        raise TypeError(f"bed_properties must be a PackedBedProperties, got {type(bed_properties).__name__}.")

    if bed_properties is None:
        system = assemble_phase_momentum_system(
            mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure, boundary_conditions, component
        )
    else:
        system = assemble_packed_bed_phase_momentum_system(
            mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure, bed_properties,
            boundary_conditions, component,
        )

    interior = _interior_mask(mesh, boundary_conditions)
    inv_dt = 1.0 / dt
    for cell_index in np.where(interior)[0]:
        cell_index = int(cell_index)
        current_diagonal = system.matrix.get(cell_index, cell_index)
        system.matrix.set(cell_index, cell_index, current_diagonal + inv_dt)

    system.rhs[interior] += inv_dt * previous_velocity.values[interior, index]

    return system


def _solve_transient_phase_velocity(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    previous_velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    dt: float,
    method: str,
    max_iterations: int,
    tolerance: float,
    bed_properties: Optional[PackedBedProperties],
) -> VectorField:
    """Solve both velocity components of one phase's transient momentum equation."""
    u_system = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_phase_velocity, drag_coefficient, pressure,
        u_boundary_conditions, "u", dt, bed_properties,
    )
    v_system = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_phase_velocity, drag_coefficient, pressure,
        v_boundary_conditions, "v", dt, bed_properties,
    )

    u_solver = _build_solver(method, max_iterations, tolerance)
    v_solver = _build_solver(method, max_iterations, tolerance)

    u_values = u_solver.solve(u_system, initial_guess=velocity.values[:, 0])
    v_values = v_solver.solve(v_system, initial_guess=velocity.values[:, 1])

    return VectorField(mesh, np.column_stack([u_values, v_values]))


def _relax(previous: VectorField, updated: VectorField, relaxation: float) -> VectorField:
    blended = relaxation * updated.values + (1.0 - relaxation) * previous.values
    return VectorField(previous.mesh, blended)


def _rms_change(previous: VectorField, updated: VectorField) -> float:
    return float(np.sqrt(np.mean((updated.values - previous.values) ** 2)))


def _mixture_velocity(
    dispersed: Phase, continuous: Phase, dispersed_velocity: VectorField, continuous_velocity: VectorField
) -> VectorField:
    """Return the mass-weighted mixture velocity for the current phase velocities.

    Mirrors :meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.mixture_velocity`,
    but uses the caller-supplied (currently evolving) phase velocities rather
    than the ``Phase`` objects' own (unchanging) ``velocity`` attribute, so it
    can be used to warm-start each time step's mixture pressure solve.
    """
    weight_dispersed = dispersed.volume_fraction.values * dispersed.density
    weight_continuous = continuous.volume_fraction.values * continuous.density
    numerator = (
        weight_dispersed[:, None] * dispersed_velocity.values
        + weight_continuous[:, None] * continuous_velocity.values
    )
    denominator = weight_dispersed + weight_continuous
    return VectorField(dispersed.mesh, numerator / denominator[:, None])


def _advance_time_step(
    mesh: Mesh,
    dispersed: Phase,
    continuous: Phase,
    dispersed_name: str,
    continuous_name: str,
    dispersed_velocity_prev: VectorField,
    continuous_velocity_prev: VectorField,
    pressure: ScalarField,
    particle_diameter: float,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    dt: float,
    velocity_relaxation: float,
    max_outer_iterations: int,
    outer_tolerance: float,
    linear_method: str,
    linear_max_iterations: int,
    linear_tolerance: float,
    bed_properties: Optional[PackedBedProperties],
) -> Tuple[VectorField, VectorField, ScalarField, int, bool, List[Dict[str, float]]]:
    """Advance both phases' velocities by one implicit time step.

    Runs the same outer (Gauss-Seidel phase-coupling, under-relaxed)
    iteration :func:`~src.cfd.eulerian_solver.solve_eulerian_two_fluid` uses
    to resolve convective/drag nonlinearity, but with
    :func:`assemble_transient_phase_momentum_system` supplying the ddt term
    referencing ``*_velocity_prev`` (U^n, held fixed for the whole step)
    rather than the steady assembly.
    """
    dispersed_velocity = VectorField(mesh, dispersed_velocity_prev.values.copy())
    continuous_velocity = VectorField(mesh, continuous_velocity_prev.values.copy())
    drag_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    residual_history: List[Dict[str, float]] = []
    converged = False
    iterations_run = 0

    for iteration in range(max_outer_iterations):
        relative_speed = relative_velocity_magnitude(dispersed_velocity, continuous_velocity)
        drag_field = interphase_drag_coefficient(
            mesh,
            continuous.density,
            continuous.viscosity,
            dispersed.volume_fraction,
            relative_speed,
            particle_diameter,
        )

        new_dispersed_velocity = _solve_transient_phase_velocity(
            mesh, dispersed, dispersed_velocity, dispersed_velocity_prev, continuous_velocity, drag_field, pressure,
            u_boundary_conditions, v_boundary_conditions, dt, linear_method, linear_max_iterations, linear_tolerance,
            bed_properties,
        )
        new_continuous_velocity = _solve_transient_phase_velocity(
            mesh, continuous, continuous_velocity, continuous_velocity_prev, dispersed_velocity, drag_field, pressure,
            u_boundary_conditions, v_boundary_conditions, dt, linear_method, linear_max_iterations, linear_tolerance,
            bed_properties,
        )

        relaxed_dispersed_velocity = _relax(dispersed_velocity, new_dispersed_velocity, velocity_relaxation)
        relaxed_continuous_velocity = _relax(continuous_velocity, new_continuous_velocity, velocity_relaxation)

        residuals = {
            f"{dispersed_name}_velocity": _rms_change(dispersed_velocity, relaxed_dispersed_velocity),
            f"{continuous_name}_velocity": _rms_change(continuous_velocity, relaxed_continuous_velocity),
        }
        residual_history.append(residuals)
        iterations_run = iteration + 1

        dispersed_velocity = relaxed_dispersed_velocity
        continuous_velocity = relaxed_continuous_velocity

        if all(value < outer_tolerance for value in residuals.values()):
            converged = True
            break

    return dispersed_velocity, continuous_velocity, drag_field, iterations_run, converged, residual_history


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class TransientEulerianTwoFluidResult:
    """Store the full time history of a transient two-fluid Euler-Euler solve.

    Every history list (``velocity_history[phase]``, ``pressure_history``,
    ``volume_fraction_history[phase]``, ``drag_coefficient_history``) has one
    entry per stored time point, including the initial condition at
    ``time_points[0] == 0.0``, so ``len(time_points) == n_steps + 1``.

    Parameters
    ----------
    time_points : list of float
        Simulation time at each stored entry, starting at ``0.0``.
    velocity_history : dict of str to list of VectorField
        Each phase's velocity field at every stored time point, keyed by
        phase name.
    pressure_history : list of ScalarField
        The shared pressure field at every stored time point.
    volume_fraction_history : dict of str to list of ScalarField
        Each phase's volume-fraction field at every stored time point, keyed
        by phase name. Volume-fraction transport is not yet implemented (see
        the module docstring), so these values are currently constant across
        time, but are stored at every step for a consistent history.
    drag_coefficient_history : list of ScalarField
        The interphase drag coefficient field, K, at every stored time point.
    dt : float
        The (positive) time step size used for the whole simulation.
    total_time : float
        The (positive) total simulation time.
    n_steps : int
        Number of time steps actually taken.
    iterations_per_step : list of int
        Number of outer (phase-coupling) iterations run at each stored time
        point (``0`` for the initial condition).
    converged_per_step : list of bool
        Whether each stored time point's outer iteration converged (always
        ``True`` for the initial condition).
    residual_history_per_step : list of list of dict
        Each stored time point's outer-iteration residual history (an empty
        list for the initial condition).
    bed_properties : PackedBedProperties, optional
        The packed-bed properties used for this solve, if any, kept so that
        :meth:`build_force_accounting` can also report the packed-bed
        resistance force.
    """

    def __init__(
        self,
        time_points: List[float],
        velocity_history: Dict[str, List[VectorField]],
        pressure_history: List[ScalarField],
        volume_fraction_history: Dict[str, List[ScalarField]],
        drag_coefficient_history: List[ScalarField],
        dt: float,
        total_time: float,
        n_steps: int,
        iterations_per_step: List[int],
        converged_per_step: List[bool],
        residual_history_per_step: List[List[Dict[str, float]]],
        bed_properties: Optional[PackedBedProperties] = None,
    ) -> None:
        self.time_points = time_points
        self.velocity_history = velocity_history
        self.pressure_history = pressure_history
        self.volume_fraction_history = volume_fraction_history
        self.drag_coefficient_history = drag_coefficient_history
        self.dt = dt
        self.total_time = total_time
        self.n_steps = n_steps
        self.iterations_per_step = iterations_per_step
        self.converged_per_step = converged_per_step
        self.residual_history_per_step = residual_history_per_step
        self.bed_properties = bed_properties

    @property
    def phase_names(self) -> List[str]:
        """Names of every phase whose history this result stores."""
        return list(self.velocity_history.keys())

    @property
    def converged(self) -> bool:
        """Whether every time step's outer iteration converged."""
        return all(self.converged_per_step)

    def velocity(self, phase_name: str, step: int = -1) -> VectorField:
        """Return the named phase's velocity field at a stored time step.

        Parameters
        ----------
        phase_name : str
            Name of the phase to look up.
        step : int
            Index into the stored history. Defaults to ``-1`` (the final
            time point).

        Raises
        ------
        KeyError
            If no phase with that name was solved.
        IndexError
            If ``step`` is out of range.
        """
        try:
            history = self.velocity_history[phase_name]
        except KeyError as exc:
            raise KeyError(f"no phase named '{phase_name}' in this result.") from exc
        try:
            return history[step]
        except IndexError as exc:
            raise IndexError(f"step {step} is out of range for {len(history)} stored time point(s).") from exc

    def pressure(self, step: int = -1) -> ScalarField:
        """Return the shared pressure field at a stored time step."""
        try:
            return self.pressure_history[step]
        except IndexError as exc:
            raise IndexError(
                f"step {step} is out of range for {len(self.pressure_history)} stored time point(s)."
            ) from exc

    def volume_fraction(self, phase_name: str, step: int = -1) -> ScalarField:
        """Return the named phase's volume-fraction field at a stored time step."""
        try:
            history = self.volume_fraction_history[phase_name]
        except KeyError as exc:
            raise KeyError(f"no phase named '{phase_name}' in this result.") from exc
        try:
            return history[step]
        except IndexError as exc:
            raise IndexError(f"step {step} is out of range for {len(history)} stored time point(s).") from exc

    def build_force_accounting(self, phase_name: str, step: int = -1) -> ForceAccounting:
        """Build a ForceAccounting report for one phase at one stored time step.

        Registers the pressure-gradient force (``-alpha_phase * grad(p)``)
        and interphase-drag force (``K * (U_other - U_phase)``), reusing
        :class:`~src.cfd.force_accounting.ForceAccounting` and
        :func:`~src.cfd.operators.gradient` rather than reimplementing force
        bookkeeping. If this result was produced with packed-bed resistance
        active, the Ergun resistance force is registered too, reusing
        :class:`~src.cfd.packed_bed.ErgunResistanceModel`.

        Parameters
        ----------
        phase_name : str
            Name of the phase to report on.
        step : int
            Index into the stored history. Defaults to ``-1`` (the final
            time point).

        Returns
        -------
        ForceAccounting
            A force accounting instance with every applicable contribution
            registered.

        Raises
        ------
        KeyError
            If no phase with that name was solved.
        ValueError
            If this result does not contain exactly two phases.
        IndexError
            If ``step`` is out of range.
        """
        if phase_name not in self.velocity_history:
            raise KeyError(f"no phase named '{phase_name}' in this result.")
        other_names = [name for name in self.velocity_history if name != phase_name]
        if len(other_names) != 1:
            raise ValueError("build_force_accounting only supports two-phase results.")
        other_name = other_names[0]

        pressure_field = self.pressure(step)
        mesh = pressure_field.mesh
        this_velocity = self.velocity(phase_name, step)
        other_velocity = self.velocity(other_name, step)
        this_volume_fraction = self.volume_fraction(phase_name, step)
        drag_coefficient = self.drag_coefficient_history[step]

        accounting = ForceAccounting(mesh)

        pressure_force = -this_volume_fraction.values[:, None] * gradient(pressure_field).values
        accounting.register("pressure_gradient", VectorField(mesh, pressure_force))

        drag_force = drag_coefficient.values[:, None] * (other_velocity.values - this_velocity.values)
        accounting.register("interphase_drag", VectorField(mesh, drag_force))

        if self.bed_properties is not None:
            model = ErgunResistanceModel(mesh, self.bed_properties)
            accounting.register("packed_bed_resistance", model.momentum_source(this_velocity))

        return accounting

    def __repr__(self) -> str:
        return (
            f"TransientEulerianTwoFluidResult(phases={self.phase_names!r}, n_steps={self.n_steps}, "
            f"dt={self.dt}, total_time={self.total_time}, converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# Main transient two-fluid solve
# ---------------------------------------------------------------------------

def solve_transient_eulerian_two_fluid(
    system: EulerianMultiphaseSystem,
    dispersed_phase: str,
    particle_diameter: float,
    dt: float,
    total_time: float,
    u_boundary_conditions: BoundaryConditionSchedule,
    v_boundary_conditions: BoundaryConditionSchedule,
    velocity_relaxation: float = 0.5,
    max_outer_iterations: int = 200,
    outer_tolerance: float = 1e-6,
    linear_method: str = "gauss_seidel",
    linear_max_iterations: int = 2000,
    linear_tolerance: float = 1e-8,
    mixture_max_outer_iterations: int = 200,
    mixture_outer_tolerance: float = 1e-6,
    bed_properties: Optional[PackedBedProperties] = None,
) -> TransientEulerianTwoFluidResult:
    """March the two-fluid (Euler-Euler) momentum equations forward in time.

    At every time step:

        1. the shared mixture pressure is re-solved from the
           volume-fraction-weighted mixture momentum equation
           (:meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.solve_mixture_flow`),
           warm-started from the previous step's mixture velocity and
           pressure;
        2. both phases' momentum equations are advanced by one first-order
           implicit (Backward Euler) time step
           (:func:`assemble_transient_phase_momentum_system`), iterated
           Gauss-Seidel style with under-relaxation until both phases'
           velocity residuals drop below ``outer_tolerance`` or
           ``max_outer_iterations`` is reached;
        3. the resulting velocity, pressure and (currently unchanging)
           volume-fraction fields are appended to this result's history.

    Parameters
    ----------
    system : EulerianMultiphaseSystem
        A system containing exactly two phases, attached to a structured,
        uniformly-spaced 2D Cartesian mesh. Each phase's ``velocity`` and
        ``pressure`` fields supply the initial condition at ``time=0``.
    dispersed_phase : str
        Name of the phase treated as the dispersed (bubble/droplet/particle)
        phase for the drag correlation. The other phase in ``system`` is
        treated as continuous.
    particle_diameter : float
        The (constant, positive) dispersed-phase particle/droplet/bubble
        diameter used by the drag correlation.
    dt : float
        The (positive) time step size.
    total_time : float
        The (positive) total simulation time. Must be an integer multiple
        of ``dt``.
    u_boundary_conditions, v_boundary_conditions : BoundaryConditionSchedule
        Either a fixed list/tuple of exactly four FixedValueBC instances
        (used unchanged at every time step - "steady" boundary conditions),
        or a callable ``time -> list of four FixedValueBC`` that is
        re-evaluated at every time step ("transient" boundary conditions,
        e.g. a ramped inlet velocity).
    velocity_relaxation : float
        Under-relaxation factor applied to each phase's momentum update
        within a time step's outer iteration, in (0, 1]. Defaults to
        ``0.5``.
    max_outer_iterations : int
        Maximum number of outer (phase-coupling) iterations per time step.
    outer_tolerance : float
        Outer iteration for a time step stops once both phases' velocity
        residuals drop below this value.
    linear_method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``, used for every inner
        linear solve.
    linear_max_iterations : int
        Maximum number of iterations for each inner linear solve.
    linear_tolerance : float
        Convergence tolerance for each inner linear solve.
    mixture_max_outer_iterations : int
        Maximum number of SIMPLE outer iterations used for the shared
        mixture pressure solve at each time step.
    mixture_outer_tolerance : float
        Outer tolerance used for the shared mixture pressure solve.
    bed_properties : PackedBedProperties, optional
        If given, Ergun-equation packed-bed resistance is added to both
        phases' momentum equations at every time step, reusing
        :func:`~src.cfd.packed_bed.assemble_packed_bed_phase_momentum_system`.

    Returns
    -------
    TransientEulerianTwoFluidResult
        The full time history of both phases' velocities, the shared
        pressure, and both phases' volume fractions, plus per-step
        convergence bookkeeping.
    """
    if not isinstance(system, EulerianMultiphaseSystem):
        raise TypeError(f"system must be an EulerianMultiphaseSystem, got {type(system).__name__}.")
    if system.n_phases != 2:
        raise ValueError("solve_transient_eulerian_two_fluid only supports two-phase (gas-liquid) systems.")

    mesh = system.mesh
    _validate_structured_mesh(mesh)

    if not isinstance(dispersed_phase, str):
        raise TypeError(f"dispersed_phase must be a string, got {type(dispersed_phase).__name__}.")
    if dispersed_phase not in system.phase_names:
        raise ValueError(f"dispersed_phase '{dispersed_phase}' is not a phase in this system.")
    continuous_phase = next(name for name in system.phase_names if name != dispersed_phase)

    particle_diameter = _validate_positive_number(particle_diameter, "particle_diameter")
    dt = _validate_positive_number(dt, "dt")
    total_time = _validate_positive_number(total_time, "total_time")
    n_steps = _count_time_steps(dt, total_time)

    velocity_relaxation = _validate_relaxation_factor(velocity_relaxation, "velocity_relaxation")
    max_outer_iterations = _validate_positive_integer(max_outer_iterations, "max_outer_iterations")
    outer_tolerance = _validate_positive_number(outer_tolerance, "outer_tolerance")
    linear_max_iterations = _validate_positive_integer(linear_max_iterations, "linear_max_iterations")
    linear_tolerance = _validate_positive_number(linear_tolerance, "linear_tolerance")
    mixture_max_outer_iterations = _validate_positive_integer(
        mixture_max_outer_iterations, "mixture_max_outer_iterations"
    )
    mixture_outer_tolerance = _validate_positive_number(mixture_outer_tolerance, "mixture_outer_tolerance")

    _validate_boundary_condition_schedule(u_boundary_conditions, "u_boundary_conditions")
    _validate_boundary_condition_schedule(v_boundary_conditions, "v_boundary_conditions")

    if bed_properties is not None and not isinstance(bed_properties, PackedBedProperties):
        raise TypeError(f"bed_properties must be a PackedBedProperties, got {type(bed_properties).__name__}.")

    dispersed = system.get_phase(dispersed_phase)
    continuous = system.get_phase(continuous_phase)

    dispersed_velocity = VectorField(mesh, dispersed.velocity.values.copy())
    continuous_velocity = VectorField(mesh, continuous.velocity.values.copy())
    pressure = ScalarField(mesh, dispersed.pressure.values.copy())

    time_points: List[float] = [0.0]
    velocity_history: Dict[str, List[VectorField]] = {
        dispersed_phase: [dispersed_velocity],
        continuous_phase: [continuous_velocity],
    }
    pressure_history: List[ScalarField] = [pressure]
    volume_fraction_history: Dict[str, List[ScalarField]] = {
        dispersed_phase: [ScalarField(mesh, dispersed.volume_fraction.values.copy())],
        continuous_phase: [ScalarField(mesh, continuous.volume_fraction.values.copy())],
    }
    drag_coefficient_history: List[ScalarField] = [ScalarField(mesh, np.zeros(mesh.n_cells))]
    iterations_per_step: List[int] = [0]
    converged_per_step: List[bool] = [True]
    residual_history_per_step: List[List[Dict[str, float]]] = [[]]

    for step in range(1, n_steps + 1):
        time = step * dt
        u_bcs = _resolve_boundary_conditions(u_boundary_conditions, time, "u_boundary_conditions")
        v_bcs = _resolve_boundary_conditions(v_boundary_conditions, time, "v_boundary_conditions")

        warm_start_velocity = _mixture_velocity(dispersed, continuous, dispersed_velocity, continuous_velocity)
        mixture_result = system.solve_mixture_flow(
            u_bcs,
            v_bcs,
            initial_velocity=warm_start_velocity,
            initial_pressure=pressure,
            max_outer_iterations=mixture_max_outer_iterations,
            outer_tolerance=mixture_outer_tolerance,
            linear_method=linear_method,
            linear_max_iterations=linear_max_iterations,
            linear_tolerance=linear_tolerance,
        )
        pressure = mixture_result.pressure

        (
            dispersed_velocity,
            continuous_velocity,
            drag_field,
            iterations_run,
            converged,
            residuals,
        ) = _advance_time_step(
            mesh,
            dispersed,
            continuous,
            dispersed_phase,
            continuous_phase,
            dispersed_velocity,
            continuous_velocity,
            pressure,
            particle_diameter,
            u_bcs,
            v_bcs,
            dt,
            velocity_relaxation,
            max_outer_iterations,
            outer_tolerance,
            linear_method,
            linear_max_iterations,
            linear_tolerance,
            bed_properties,
        )

        time_points.append(time)
        velocity_history[dispersed_phase].append(dispersed_velocity)
        velocity_history[continuous_phase].append(continuous_velocity)
        pressure_history.append(pressure)
        volume_fraction_history[dispersed_phase].append(ScalarField(mesh, dispersed.volume_fraction.values.copy()))
        volume_fraction_history[continuous_phase].append(
            ScalarField(mesh, continuous.volume_fraction.values.copy())
        )
        drag_coefficient_history.append(drag_field)
        iterations_per_step.append(iterations_run)
        converged_per_step.append(converged)
        residual_history_per_step.append(residuals)

    return TransientEulerianTwoFluidResult(
        time_points=time_points,
        velocity_history=velocity_history,
        pressure_history=pressure_history,
        volume_fraction_history=volume_fraction_history,
        drag_coefficient_history=drag_coefficient_history,
        dt=dt,
        total_time=total_time,
        n_steps=n_steps,
        iterations_per_step=iterations_per_step,
        converged_per_step=converged_per_step,
        residual_history_per_step=residual_history_per_step,
        bed_properties=bed_properties,
    )
