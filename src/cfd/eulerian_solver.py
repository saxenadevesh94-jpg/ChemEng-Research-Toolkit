"""Educational two-fluid (Euler-Euler) solver for structured 2D gas-liquid CFD.

Sprint 27 extends the Eulerian multiphase framework built in
:mod:`~src.cfd.multiphase` from a single shared mixture-momentum equation to
the classic *two-fluid model*: one momentum equation per phase, coupled
through interphase drag rather than collapsed into one shared-velocity
mixture equation.

    alpha_k * rho_k * (U_k . grad) U_k =
        -alpha_k * grad(p) + alpha_k * mu_k * laplacian(U_k)
        + K * (U_other - U_k)

for a dispersed phase (bubbles/droplets/particles of diameter
``particle_diameter``) suspended in a continuous phase, where ``K`` is the
interphase momentum exchange coefficient given by the Schiller-Naumann drag
correlation. Dividing through by ``alpha_k * rho_k`` turns this into a
convection-diffusion equation with two extra source terms:

    (U_k . grad) U_k = -gradient(p)/rho_k + (mu_k/rho_k) * laplacian(U_k)
                        + K * (U_other - U_k) / (alpha_k * rho_k)

Both phases are assumed to share one pressure field, obtained by first
solving the existing volume-fraction-weighted mixture momentum equation via
:meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.solve_mixture_flow`
(itself built on :func:`~src.cfd.navier_stokes.solve_navier_stokes` and
:func:`~src.cfd.simple_solver.solve_simple`). Each phase's own momentum
equation is then solved as a convection-diffusion problem, reusing
:func:`~src.cfd.convection_diffusion.assemble_convection_diffusion_system`
for the convection/diffusion terms, with the shared pressure gradient and an
implicit drag term added as source terms. The two phases are iterated
(Gauss-Seidel style, one after the other) with under-relaxation until both
velocities stop changing.

It reuses the same Mesh, ScalarField, VectorField, BoundaryCondition,
Equation and LinearSystem infrastructure built in earlier sprints, plus the
Navier-Stokes/SIMPLE solver and Eulerian multiphase framework, rather than
reimplementing any of it. No separate interphase drag module exists yet in
this package, so the Schiller-Naumann correlation is implemented here,
alongside the momentum coupling that consumes it.

Only structured, uniformly-spaced 2D Cartesian meshes and exactly two-phase
(gas-liquid) systems are supported, matching the restrictions already placed
on :func:`~src.cfd.simple_solver.solve_simple`. Boundary conditions apply
identically to the mixture solve and to both phases' momentum equations,
matching the single ``u_boundary_conditions``/``v_boundary_conditions``
convention already used by :func:`~src.cfd.navier_stokes.solve_navier_stokes`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .convection_diffusion import assemble_convection_diffusion_system
from .equation import Equation
from .field import ScalarField, VectorField
from .linear_system import LinearSystem
from .mesh import Mesh
from .multiphase import EulerianMultiphaseSystem, Phase
from .navier_stokes import NavierStokesResult
from .operators import gradient
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}
_COMPONENT_INDEX = {"u": 0, "v": 1}

# Below this Reynolds number, Schiller-Naumann's power-law correction is the
# active branch; above it, the correlation is capped at the Newton-regime
# constant of 0.44, matching the correlation's usual validity range
# (Re_p <~ 1000 for a single rigid/fluid sphere).
_SCHILLER_NAUMANN_TRANSITION_REYNOLDS = 1000.0
_NEWTON_REGIME_DRAG_COEFFICIENT = 0.44

# A relative velocity of exactly zero would divide by zero when forming the
# particle Reynolds number; flooring it at a tiny value instead reproduces
# the (finite) Stokes-drag limit, since Cd * |U_r| -> 24 * mu_c / (rho_c * d_p)
# as |U_r| -> 0.
_MIN_RELATIVE_VELOCITY = 1e-12

# A dispersed-phase volume fraction of exactly zero would divide by zero when
# forming the drag term's alpha_k * rho_k normalisation; flooring it keeps
# the momentum system well-posed even where a phase is locally absent.
_MIN_VOLUME_FRACTION = 1e-6


# ---------------------------------------------------------------------------
# Descriptive bookkeeping
# ---------------------------------------------------------------------------

def phase_momentum_equation(phase_name: str, other_phase_name: str) -> Equation:
    """Return an Equation describing one phase's momentum equation in the two-fluid model.

    This is purely descriptive bookkeeping — it documents the PDE
    :func:`solve_eulerian_two_fluid` actually assembles and solves, using the
    same symbolic Equation container the rest of the CFD package uses.

    Parameters
    ----------
    phase_name : str
        Name of the phase this equation describes.
    other_phase_name : str
        Name of the other phase it exchanges momentum with through drag.

    Returns
    -------
    Equation
        ``convection(U_<phase_name>) = -gradient(p) / rho_<phase_name> +
        viscosity_<phase_name> * laplacian(U_<phase_name>) +
        drag(U_<other_phase_name> - U_<phase_name>) / (alpha_<phase_name> * rho_<phase_name>)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    if not isinstance(other_phase_name, str) or not other_phase_name.strip():
        raise ValueError("other_phase_name must be a non-empty string.")
    return Equation(
        lhs=f"convection(U_{phase_name})",
        rhs=(
            f"-gradient(p) / rho_{phase_name} "
            f"+ viscosity_{phase_name} * laplacian(U_{phase_name}) "
            f"+ drag(U_{other_phase_name} - U_{phase_name}) / (alpha_{phase_name} * rho_{phase_name})"
        ),
    )


# ---------------------------------------------------------------------------
# Schiller-Naumann interphase drag model
# ---------------------------------------------------------------------------

def schiller_naumann_drag_coefficient(reynolds_number: np.ndarray) -> np.ndarray:
    """Evaluate the Schiller-Naumann drag coefficient correlation.

    Parameters
    ----------
    reynolds_number : numpy.ndarray
        Particle Reynolds number(s). Must be non-negative.

    Returns
    -------
    numpy.ndarray
        ``Cd = 24/Re * (1 + 0.15 * Re**0.687)`` for ``Re <= 1000``, else the
        constant Newton-regime value ``0.44``.

    Raises
    ------
    ValueError
        If any Reynolds number is negative.
    """
    reynolds = np.asarray(reynolds_number, dtype=float)
    if np.any(reynolds < 0.0):
        raise ValueError("reynolds_number must be non-negative.")

    laminar_branch = (24.0 / reynolds) * (1.0 + 0.15 * reynolds ** 0.687)
    return np.where(
        reynolds <= _SCHILLER_NAUMANN_TRANSITION_REYNOLDS, laminar_branch, _NEWTON_REGIME_DRAG_COEFFICIENT
    )


def relative_velocity_magnitude(velocity_a: VectorField, velocity_b: VectorField) -> ScalarField:
    """Return the per-cell magnitude of ``velocity_a - velocity_b``.

    Parameters
    ----------
    velocity_a, velocity_b : VectorField
        Two velocity fields attached to the same mesh, each with two
        components.

    Returns
    -------
    ScalarField
        ``|velocity_a - velocity_b|`` at every cell.
    """
    _validate_velocity_field(velocity_a, "velocity_a")
    _validate_velocity_field(velocity_b, "velocity_b")
    if velocity_a.mesh is not velocity_b.mesh:
        raise ValueError("velocity_a and velocity_b must be attached to the same mesh.")

    difference = velocity_a.values - velocity_b.values
    magnitude = np.sqrt(np.sum(difference ** 2, axis=1))
    return ScalarField(velocity_a.mesh, magnitude)


def interphase_drag_coefficient(
    mesh: Mesh,
    continuous_density: float,
    continuous_viscosity: float,
    dispersed_volume_fraction: ScalarField,
    relative_velocity: ScalarField,
    particle_diameter: float,
) -> ScalarField:
    """Return the interphase momentum exchange coefficient field, K.

    Uses the Schiller-Naumann drag correlation for a dispersed phase of
    spherical particles/droplets/bubbles of diameter ``particle_diameter``
    suspended in a continuous phase:

        Re_p = rho_c * |U_r| * d_p / mu_c
        Cd   = schiller_naumann_drag_coefficient(Re_p)
        K    = 0.75 * Cd * alpha_d * rho_c * |U_r| / d_p

    ``K`` is the same for both phases (Newton's third law): it multiplies
    ``(U_other - U_k)`` in each phase's own momentum equation.

    Parameters
    ----------
    mesh : Mesh
        The mesh every field below is attached to.
    continuous_density, continuous_viscosity : float
        The (constant, positive) density and dynamic viscosity of the
        continuous phase.
    dispersed_volume_fraction : ScalarField
        The dispersed phase's volume fraction, attached to ``mesh``.
    relative_velocity : ScalarField
        ``|U_dispersed - U_continuous|`` at every cell, attached to ``mesh``
        (see :func:`relative_velocity_magnitude`).
    particle_diameter : float
        The (constant, positive) dispersed-phase particle/droplet/bubble
        diameter.

    Returns
    -------
    ScalarField
        The interphase drag coefficient, K, at every cell.
    """
    _validate_mesh_type(mesh)
    continuous_density = _validate_positive_number(continuous_density, "continuous_density")
    continuous_viscosity = _validate_positive_number(continuous_viscosity, "continuous_viscosity")
    particle_diameter = _validate_positive_number(particle_diameter, "particle_diameter")
    _validate_scalar_field(dispersed_volume_fraction, mesh, "dispersed_volume_fraction")
    _validate_scalar_field(relative_velocity, mesh, "relative_velocity")

    relative_speed = np.maximum(relative_velocity.values, _MIN_RELATIVE_VELOCITY)
    reynolds = continuous_density * relative_speed * particle_diameter / continuous_viscosity
    drag_coefficient = schiller_naumann_drag_coefficient(reynolds)

    exchange_coefficient = (
        0.75
        * drag_coefficient
        * dispersed_volume_fraction.values
        * continuous_density
        * relative_speed
        / particle_diameter
    )
    return ScalarField(mesh, exchange_coefficient)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_mesh_type(mesh: Mesh) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")


def _validate_structured_mesh(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencils used by momentum assembly assume every interior
    point has a left/right/top/bottom neighbour at a constant spacing, so we
    verify that up front rather than failing confusingly deep inside
    assembly.
    """
    _validate_mesh_type(mesh)
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("solve_eulerian_two_fluid only supports 2D Cartesian meshes.")

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


def _validate_scalar_field(field: object, mesh: Mesh, name: str) -> None:
    if not isinstance(field, ScalarField):
        raise TypeError(f"{name} must be a ScalarField, got {type(field).__name__}.")
    if field.mesh is not mesh:
        raise ValueError(f"{name} must be attached to the same mesh being solved.")


def _validate_velocity_boundary_conditions(boundary_conditions: Sequence[BoundaryCondition], name: str) -> None:
    """Check that exactly one FixedValueBC is given for each of the four sides.

    Momentum assembly only supports Dirichlet velocity boundaries, so any
    other BoundaryCondition subclass — such as ZeroGradientBC — is rejected
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


# ---------------------------------------------------------------------------
# Per-phase momentum assembly
# ---------------------------------------------------------------------------

def assemble_phase_momentum_system(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    boundary_conditions: Sequence[BoundaryCondition],
    component: str,
) -> LinearSystem:
    """Assemble one velocity component's momentum system for a two-fluid phase.

    Reuses :func:`~src.cfd.convection_diffusion.assemble_convection_diffusion_system`
    for the convection and (kinematic) diffusion terms, then adds two source
    terms to every interior row:

        * the shared pressure gradient, ``-gradient(p) / rho_phase``
        * an implicit interphase drag term, added to both the matrix
          diagonal and the right-hand side so that ``other_phase_velocity``
          pulls this phase's velocity towards it rather than overshooting:

              (a_P + K/(alpha*rho)) * phi_P + ... =
                  ... + K/(alpha*rho) * other_phase_velocity

    Boundary rows are left untouched (they already pin phi to its
    FixedValueBC value, independent of pressure or drag).

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    phase : Phase
        The phase this momentum equation is being assembled for. Its
        ``density``, ``viscosity`` and ``volume_fraction`` supply the
        equation's material properties.
    velocity : VectorField
        The current velocity guess for ``phase``, used to evaluate the
        convective term.
    other_phase_velocity : VectorField
        The current velocity guess for the other phase, used by the
        (implicit) drag source term.
    drag_coefficient : ScalarField
        The interphase momentum exchange coefficient, K, from
        :func:`interphase_drag_coefficient`.
    pressure : ScalarField
        The shared pressure field.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances for this velocity component.
    component : str
        Either ``"u"`` (uses d(p)/dx) or ``"v"`` (uses d(p)/dy).

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for this component.
    """
    index = _validate_component(component)
    _validate_structured_mesh(mesh)
    if not isinstance(phase, Phase):
        raise TypeError(f"phase must be a Phase instance, got {type(phase).__name__}.")
    _validate_velocity_field(velocity, "velocity")
    _validate_velocity_field(other_phase_velocity, "other_phase_velocity")
    _validate_scalar_field(drag_coefficient, mesh, "drag_coefficient")
    _validate_scalar_field(pressure, mesh, "pressure")
    _validate_velocity_boundary_conditions(boundary_conditions, "boundary_conditions")

    kinematic_viscosity = phase.viscosity / phase.density
    system = assemble_convection_diffusion_system(mesh, velocity, boundary_conditions, kinematic_viscosity)

    pressure_gradient = gradient(pressure).values[:, index] / phase.density
    alpha = np.maximum(phase.volume_fraction.values, _MIN_VOLUME_FRACTION)
    drag_scale = drag_coefficient.values / (alpha * phase.density)

    interior = _interior_mask(mesh, boundary_conditions)
    for cell_index in np.where(interior)[0]:
        cell_index = int(cell_index)
        current_diagonal = system.matrix.get(cell_index, cell_index)
        system.matrix.set(cell_index, cell_index, current_diagonal + drag_scale[cell_index])

    system.rhs[interior] += (
        drag_scale[interior] * other_phase_velocity.values[interior, index] - pressure_gradient[interior]
    )

    return system


def _solve_phase_velocity(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    method: str,
    max_iterations: int,
    tolerance: float,
) -> VectorField:
    """Solve both velocity components of one phase's momentum equation."""
    u_system = assemble_phase_momentum_system(
        mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure, u_boundary_conditions, "u"
    )
    v_system = assemble_phase_momentum_system(
        mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure, v_boundary_conditions, "v"
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


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class EulerianTwoFluidResult:
    """Container for the outcome of a two-fluid Euler-Euler solve.

    Parameters
    ----------
    velocities : dict of str to VectorField
        Each phase's final velocity field, keyed by phase name.
    pressure : ScalarField
        The shared pressure field (from the mixture-momentum solve).
    drag_coefficient : ScalarField
        The final interphase momentum exchange coefficient field, K.
    mixture_result : NavierStokesResult
        The underlying mixture-momentum solve that produced ``pressure``.
    iterations_run : int
        Number of outer (phase-coupling) iterations actually run.
    converged : bool
        Whether both phases' velocity residuals dropped below the requested
        tolerance.
    residual_history : list of dict
        One residual dictionary per outer iteration, keyed
        ``"<phase_name>_velocity"``.
    """

    def __init__(
        self,
        velocities: Dict[str, VectorField],
        pressure: ScalarField,
        drag_coefficient: ScalarField,
        mixture_result: NavierStokesResult,
        iterations_run: int,
        converged: bool,
        residual_history: List[Dict[str, float]],
    ) -> None:
        self.velocities = velocities
        self.pressure = pressure
        self.drag_coefficient = drag_coefficient
        self.mixture_result = mixture_result
        self.iterations_run = iterations_run
        self.converged = converged
        self.residual_history = residual_history

    def velocity(self, phase_name: str) -> VectorField:
        """Return the final velocity field for the named phase.

        Raises
        ------
        KeyError
            If no phase with that name was solved.
        """
        try:
            return self.velocities[phase_name]
        except KeyError as exc:
            raise KeyError(f"no phase named '{phase_name}' in this result.") from exc

    def __repr__(self) -> str:
        final_residual = self.residual_history[-1] if self.residual_history else None
        return (
            f"EulerianTwoFluidResult(phases={sorted(self.velocities)!r}, "
            f"iterations_run={self.iterations_run}, converged={self.converged}, "
            f"final_residual={final_residual})"
        )


# ---------------------------------------------------------------------------
# Main two-fluid solve
# ---------------------------------------------------------------------------

def solve_eulerian_two_fluid(
    system: EulerianMultiphaseSystem,
    dispersed_phase: str,
    particle_diameter: float,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    velocity_relaxation: float = 0.5,
    max_outer_iterations: int = 200,
    outer_tolerance: float = 1e-6,
    linear_method: str = "gauss_seidel",
    linear_max_iterations: int = 2000,
    linear_tolerance: float = 1e-8,
    mixture_max_outer_iterations: int = 200,
    mixture_outer_tolerance: float = 1e-6,
) -> EulerianTwoFluidResult:
    """Solve the coupled two-fluid (Euler-Euler) momentum equations for a gas-liquid system.

    Each outer iteration:

        1. computes the interphase drag coefficient field, K, from the
           current relative velocity between the two phases (Schiller-Naumann
           correlation, see :func:`interphase_drag_coefficient`);
        2. solves the dispersed phase's momentum equation using the
           continuous phase's current velocity as the (implicit) drag
           partner;
        3. solves the continuous phase's momentum equation using the
           dispersed phase's just-updated velocity (Gauss-Seidel style);
        4. under-relaxes both updates and checks convergence.

    The pressure field driving both phases is shared and solved once,
    up front, from the volume-fraction-weighted mixture momentum equation
    (:meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.solve_mixture_flow`),
    which is itself the existing SIMPLE/Navier-Stokes solver applied with
    mixture density and viscosity. No pressure-correction logic specific to
    the two-fluid model is implemented here.

    Parameters
    ----------
    system : EulerianMultiphaseSystem
        A system containing exactly two phases (e.g. ``"gas"`` and
        ``"liquid"``), attached to a structured, uniformly-spaced 2D
        Cartesian mesh.
    dispersed_phase : str
        Name of the phase treated as the dispersed (bubble/droplet/particle)
        phase for the drag correlation. The other phase in ``system`` is
        treated as continuous.
    particle_diameter : float
        The (constant, positive) dispersed-phase particle/droplet/bubble
        diameter used by the drag correlation.
    u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances each, one per boundary ("left",
        "right", "top", "bottom"). These are applied identically to the
        mixture pressure solve and to both phases' momentum equations.
    velocity_relaxation : float
        Under-relaxation factor applied to each phase's momentum update, in
        (0, 1]. Defaults to ``0.5``.
    max_outer_iterations : int
        Maximum number of outer (phase-coupling) iterations.
    outer_tolerance : float
        Outer iteration stops once both phases' velocity residuals drop
        below this value.
    linear_method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``, used for every inner
        linear solve (mixture SIMPLE loop and both phases' momentum
        prediction).
    linear_max_iterations : int
        Maximum number of iterations for each inner linear solve.
    linear_tolerance : float
        Convergence tolerance for each inner linear solve.
    mixture_max_outer_iterations : int
        Maximum number of SIMPLE outer iterations used for the shared
        mixture pressure solve.
    mixture_outer_tolerance : float
        Outer tolerance used for the shared mixture pressure solve.

    Returns
    -------
    EulerianTwoFluidResult
        The final per-phase velocities, shared pressure field, final drag
        coefficient field, and convergence bookkeeping.
    """
    if not isinstance(system, EulerianMultiphaseSystem):
        raise TypeError(f"system must be an EulerianMultiphaseSystem, got {type(system).__name__}.")
    if system.n_phases != 2:
        raise ValueError("solve_eulerian_two_fluid only supports two-phase (gas-liquid) systems.")

    mesh = system.mesh
    _validate_structured_mesh(mesh)

    if not isinstance(dispersed_phase, str):
        raise TypeError(f"dispersed_phase must be a string, got {type(dispersed_phase).__name__}.")
    if dispersed_phase not in system.phase_names:
        raise ValueError(f"dispersed_phase '{dispersed_phase}' is not a phase in this system.")
    continuous_phase = next(name for name in system.phase_names if name != dispersed_phase)

    particle_diameter = _validate_positive_number(particle_diameter, "particle_diameter")
    velocity_relaxation = _validate_relaxation_factor(velocity_relaxation, "velocity_relaxation")
    max_outer_iterations = _validate_positive_integer(max_outer_iterations, "max_outer_iterations")
    outer_tolerance = _validate_positive_number(outer_tolerance, "outer_tolerance")
    _validate_velocity_boundary_conditions(u_boundary_conditions, "u_boundary_conditions")
    _validate_velocity_boundary_conditions(v_boundary_conditions, "v_boundary_conditions")

    mixture_result = system.solve_mixture_flow(
        u_boundary_conditions,
        v_boundary_conditions,
        max_outer_iterations=mixture_max_outer_iterations,
        outer_tolerance=mixture_outer_tolerance,
        linear_method=linear_method,
        linear_max_iterations=linear_max_iterations,
        linear_tolerance=linear_tolerance,
    )
    pressure = mixture_result.pressure

    dispersed = system.get_phase(dispersed_phase)
    continuous = system.get_phase(continuous_phase)

    dispersed_velocity = VectorField(mesh, dispersed.velocity.values.copy())
    continuous_velocity = VectorField(mesh, continuous.velocity.values.copy())

    residual_history: List[Dict[str, float]] = []
    converged = False
    iterations_run = 0
    drag_field = ScalarField(mesh, np.zeros(mesh.n_cells))

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

        new_dispersed_velocity = _solve_phase_velocity(
            mesh,
            dispersed,
            dispersed_velocity,
            continuous_velocity,
            drag_field,
            pressure,
            u_boundary_conditions,
            v_boundary_conditions,
            linear_method,
            linear_max_iterations,
            linear_tolerance,
        )
        new_continuous_velocity = _solve_phase_velocity(
            mesh,
            continuous,
            continuous_velocity,
            dispersed_velocity,
            drag_field,
            pressure,
            u_boundary_conditions,
            v_boundary_conditions,
            linear_method,
            linear_max_iterations,
            linear_tolerance,
        )

        relaxed_dispersed_velocity = _relax(dispersed_velocity, new_dispersed_velocity, velocity_relaxation)
        relaxed_continuous_velocity = _relax(continuous_velocity, new_continuous_velocity, velocity_relaxation)

        residuals = {
            f"{dispersed_phase}_velocity": _rms_change(dispersed_velocity, relaxed_dispersed_velocity),
            f"{continuous_phase}_velocity": _rms_change(continuous_velocity, relaxed_continuous_velocity),
        }
        residual_history.append(residuals)
        iterations_run = iteration + 1

        dispersed_velocity = relaxed_dispersed_velocity
        continuous_velocity = relaxed_continuous_velocity

        if all(value < outer_tolerance for value in residuals.values()):
            converged = True
            break

    velocities = {dispersed_phase: dispersed_velocity, continuous_phase: continuous_velocity}
    return EulerianTwoFluidResult(
        velocities=velocities,
        pressure=pressure,
        drag_coefficient=drag_field,
        mixture_result=mixture_result,
        iterations_run=iterations_run,
        converged=converged,
        residual_history=residual_history,
    )
