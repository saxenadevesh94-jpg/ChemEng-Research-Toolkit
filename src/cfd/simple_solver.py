"""Educational SIMPLE algorithm for steady, incompressible 2D flow.

SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) is the classic
iterative scheme for solving the coupled steady incompressible Navier-Stokes
equations:

    u * d(u)/dx + v * d(u)/dy = -d(p)/dx + viscosity * laplacian(u)
    u * d(v)/dx + v * d(v)/dy = -d(p)/dy + viscosity * laplacian(v)
    d(u)/dx + d(v)/dy = 0                                              (continuity)

on a structured 2D Cartesian mesh with FixedValueBC (Dirichlet) velocity
boundaries on all four sides. Each outer iteration repeats five steps:

    1. momentum prediction  - solve the (upwind) convection-diffusion
       equation for each velocity component with the current pressure
       field supplying a source term, giving an intermediate velocity_star
       that generally does not satisfy continuity.
    2. pressure correction  - solve laplacian(p_prime) = divergence(velocity_star) / scale
       for the pressure correction that removes that divergence.
    3. velocity correction  - project velocity_star onto a divergence-free
       field using the gradient of p_prime, then re-apply the Dirichlet
       velocity boundaries.
    4. residual computation - measure how much the continuity and momentum
       equations still disagree with the corrected fields.
    5. convergence checking - stop once every residual drops below the
       requested tolerance.

It reuses the same Mesh, ScalarField, VectorField, BoundaryCondition,
Equation, LinearSystem, JacobiSolver/GaussSeidelSolver, the convection-
diffusion assembly (for momentum prediction) and the pressure-correction
assembly (for the pressure-velocity coupling) built in earlier sprints
rather than reimplementing any of it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC, ZeroGradientBC
from .convection_diffusion import assemble_convection_diffusion_system
from .equation import Equation
from .field import ScalarField, VectorField
from .linear_system import LinearSystem
from .mesh import Mesh
from .operators import divergence, gradient
from .pressure_correction import solve_pressure_correction
from .solvers import GaussSeidelSolver, JacobiSolver

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}

# Corner cells sit on two boundaries at once (e.g. top-left is on both "left"
# and "top"). Applying BCs in this fixed order - rather than the order the
# caller happened to list them in - makes the outcome deterministic: top/
# bottom are applied last, so a corner's value always comes from its
# top/bottom boundary, matching the convention used by the other CFD solvers
# in this package (e.g. lid-driven cavity flows, where the moving lid is the
# top boundary and must win at the corners).
_BOUNDARY_APPLICATION_ORDER = ("left", "right", "top", "bottom")

_COMPONENT_INDEX = {"u": 0, "v": 1}


def simple_momentum_equation() -> Equation:
    """Return an Equation describing the momentum prediction step.

    Returns
    -------
    Equation
        ``convection(U) = viscosity * laplacian(U) - gradient(p)``
    """
    return Equation(lhs="convection(U)", rhs="viscosity * laplacian(U) - gradient(p)")


def simple_continuity_equation() -> Equation:
    """Return an Equation describing the continuity constraint SIMPLE enforces.

    Returns
    -------
    Equation
        ``divergence(U) = 0``
    """
    return Equation(lhs="divergence(U)", rhs="0")


class SimpleSolverResult:
    """Container for the outcome of a SIMPLE outer-iteration run.

    Mirrors the small ``iterations_run`` / ``converged`` / ``residual_history``
    bookkeeping already used by :class:`~src.cfd.solvers.JacobiSolver` and
    :class:`~src.cfd.solvers.GaussSeidelSolver`.

    Parameters
    ----------
    velocity : VectorField
        The final velocity field.
    pressure : ScalarField
        The final pressure field.
    iterations_run : int
        Number of outer SIMPLE iterations actually run.
    converged : bool
        Whether every residual dropped below the requested tolerance.
    residual_history : list of dict
        One residual dictionary (see :func:`compute_residuals`) per outer
        iteration.
    """

    def __init__(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        iterations_run: int,
        converged: bool,
        residual_history: List[Dict[str, float]],
    ) -> None:
        self.velocity = velocity
        self.pressure = pressure
        self.iterations_run = iterations_run
        self.converged = converged
        self.residual_history = residual_history

    def __repr__(self) -> str:
        final_residual = self.residual_history[-1] if self.residual_history else None
        return (
            f"SimpleSolverResult(iterations_run={self.iterations_run}, "
            f"converged={self.converged}, final_residual={final_residual})"
        )


def _validate_mesh_for_simple(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid.

    The five-point stencils used by momentum prediction and pressure
    correction assume every interior point has a left/right/top/bottom
    neighbour at a constant spacing, so we verify that up front rather than
    failing confusingly deep inside assembly.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("simple_solver only supports 2D Cartesian meshes.")

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


def _validate_velocity_boundary_conditions(boundary_conditions: Sequence[BoundaryCondition], name: str) -> None:
    """Check that exactly one FixedValueBC is given for each of the four sides.

    SIMPLE's momentum prediction only supports Dirichlet velocity boundaries,
    so any other BoundaryCondition subclass - such as ZeroGradientBC - is
    rejected here rather than producing an incorrectly assembled system.
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


def _validate_pressure_field(mesh: Mesh, pressure: ScalarField) -> None:
    if not isinstance(pressure, ScalarField):
        raise TypeError(f"pressure must be a ScalarField, got {type(pressure).__name__}.")
    if pressure.mesh is not mesh:
        raise ValueError("pressure must be attached to the same mesh being solved.")


def _validate_velocity_field(mesh: Mesh, velocity: VectorField, name: str) -> None:
    if not isinstance(velocity, VectorField):
        raise TypeError(f"{name} must be a VectorField, got {type(velocity).__name__}.")
    if velocity.mesh is not mesh:
        raise ValueError(f"{name} must be attached to the same mesh being solved.")
    if velocity.values.shape[1] != 2:
        raise ValueError(f"{name} must have exactly two components (u, v).")


def _validate_or_default_velocity(mesh: Mesh, initial_velocity: Optional[VectorField]) -> VectorField:
    if initial_velocity is None:
        return VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    _validate_velocity_field(mesh, initial_velocity, "initial_velocity")
    return VectorField(mesh, initial_velocity.values.copy())


def _validate_or_default_pressure(mesh: Mesh, initial_pressure: Optional[ScalarField]) -> ScalarField:
    if initial_pressure is None:
        return ScalarField(mesh, np.zeros(mesh.n_cells))
    _validate_pressure_field(mesh, initial_pressure)
    return ScalarField(mesh, initial_pressure.values.copy())


def _build_solver(method: str, max_iterations: int, tolerance: float):
    if method == "jacobi":
        return JacobiSolver(max_iterations=max_iterations, tolerance=tolerance)
    if method == "gauss_seidel":
        return GaussSeidelSolver(max_iterations=max_iterations, tolerance=tolerance)
    raise ValueError("method must be either 'jacobi' or 'gauss_seidel'.")


# ---------------------------------------------------------------------------
# Step 1: momentum prediction
# ---------------------------------------------------------------------------

def assemble_momentum_system(
    mesh: Mesh,
    velocity: VectorField,
    boundary_conditions: Sequence[BoundaryCondition],
    viscosity: float,
    pressure: ScalarField,
    component: str,
) -> LinearSystem:
    """Assemble the linear system for one velocity component's momentum equation.

    This reuses :func:`~src.cfd.convection_diffusion.assemble_convection_diffusion_system`
    for the convection and diffusion terms, then adds the pressure-gradient
    source term to every interior row:

        convection(phi) - viscosity * laplacian(phi) = -d(p)/d(component)

    Boundary rows are left untouched (they already pin phi to its FixedValueBC
    value, independent of pressure).

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    velocity : VectorField
        The current velocity guess used to evaluate the convective term.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances for this velocity component.
    viscosity : float
        The (constant, positive) kinematic viscosity.
    pressure : ScalarField
        The current pressure field, attached to ``mesh``.
    component : str
        Either ``"u"`` (uses d(p)/dx) or ``"v"`` (uses d(p)/dy).

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for this component.
    """
    index = _validate_component(component)
    _validate_mesh_for_simple(mesh)
    _validate_pressure_field(mesh, pressure)

    system = assemble_convection_diffusion_system(mesh, velocity, boundary_conditions, viscosity)

    pressure_gradient = gradient(pressure).values[:, index]
    interior = _interior_mask(mesh, boundary_conditions)
    system.rhs[interior] -= pressure_gradient[interior]

    return system


def solve_momentum_predictor(
    mesh: Mesh,
    velocity: VectorField,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    viscosity: float,
    pressure: ScalarField,
    method: str = "gauss_seidel",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> Tuple[VectorField, LinearSystem, LinearSystem]:
    """Solve the momentum prediction step for both velocity components.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    velocity : VectorField
        The current velocity guess used to evaluate the convective term.
    u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances each, one per boundary.
    viscosity : float
        The (constant, positive) kinematic viscosity.
    pressure : ScalarField
        The current pressure field, attached to ``mesh``.
    method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``.
    max_iterations : int
        Maximum number of linear solver iterations per component.
    tolerance : float
        Linear solver convergence tolerance.

    Returns
    -------
    Tuple[VectorField, LinearSystem, LinearSystem]
        The intermediate velocity_star field, and the assembled u and v
        momentum systems (their diagonals are reused by
        :func:`momentum_relaxation_scale`).
    """
    _validate_mesh_for_simple(mesh)
    _validate_velocity_field(mesh, velocity, "velocity")
    _validate_velocity_boundary_conditions(u_boundary_conditions, "u_boundary_conditions")
    _validate_velocity_boundary_conditions(v_boundary_conditions, "v_boundary_conditions")

    u_system = assemble_momentum_system(mesh, velocity, u_boundary_conditions, viscosity, pressure, "u")
    v_system = assemble_momentum_system(mesh, velocity, v_boundary_conditions, viscosity, pressure, "v")

    u_solver = _build_solver(method, max_iterations, tolerance)
    v_solver = _build_solver(method, max_iterations, tolerance)

    u_values = u_solver.solve(u_system, initial_guess=velocity.values[:, 0])
    v_values = v_solver.solve(v_system, initial_guess=velocity.values[:, 1])

    velocity_star = VectorField(mesh, np.column_stack([u_values, v_values]))
    return velocity_star, u_system, v_system


def momentum_relaxation_scale(
    mesh: Mesh,
    u_system: LinearSystem,
    v_system: LinearSystem,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
) -> float:
    """Return the pressure-velocity coupling scale implied by the momentum systems.

    SIMPLE relates the pressure correction to the velocity correction through
    the reciprocal of the momentum equation's central ("a_P") coefficient.
    This averages that coefficient over the interior rows of both momentum
    systems and returns its reciprocal, which is used both as the ``dt`` of
    :func:`~src.cfd.pressure_correction.solve_pressure_correction` and as the
    scale applied to the pressure-correction gradient in
    :func:`correct_velocity`.

    Parameters
    ----------
    mesh : Mesh
        The mesh the momentum systems were assembled on.
    u_system, v_system : LinearSystem
        The assembled u- and v-momentum systems from
        :func:`solve_momentum_predictor`.
    u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
        The boundary conditions used to assemble ``u_system`` and
        ``v_system``, needed to identify interior rows.

    Returns
    -------
    float
        The (positive) pressure-velocity coupling scale.
    """
    if not isinstance(u_system, LinearSystem) or not isinstance(v_system, LinearSystem):
        raise TypeError("u_system and v_system must be LinearSystem instances.")
    if u_system.size != mesh.n_cells or v_system.size != mesh.n_cells:
        raise ValueError("u_system and v_system must have one row per mesh cell.")

    u_diagonal = _mean_interior_diagonal(u_system, mesh, u_boundary_conditions)
    v_diagonal = _mean_interior_diagonal(v_system, mesh, v_boundary_conditions)
    mean_diagonal = 0.5 * (u_diagonal + v_diagonal)

    if mean_diagonal <= 0:
        raise ValueError("momentum system diagonal coefficients must be positive.")

    return 1.0 / mean_diagonal


def _mean_interior_diagonal(system: LinearSystem, mesh: Mesh, boundary_conditions: Sequence[BoundaryCondition]) -> float:
    mask = _interior_mask(mesh, boundary_conditions)
    diagonal = np.array([system.matrix.get(i, i) for i in range(mesh.n_cells)])
    return float(np.mean(diagonal[mask]))


# ---------------------------------------------------------------------------
# Step 2: pressure correction
# ---------------------------------------------------------------------------

def solve_pressure_correction_step(
    mesh: Mesh,
    velocity_star: VectorField,
    relaxation_scale: float,
    method: str = "gauss_seidel",
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> ScalarField:
    """Solve the pressure correction equation for the current velocity_star.

    A thin wrapper around
    :func:`~src.cfd.pressure_correction.solve_pressure_correction` that
    always uses homogeneous Neumann (zero-gradient) pressure-correction
    boundaries on all four sides, as required by the SIMPLE algorithm.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    velocity_star : VectorField
        The intermediate velocity field from the momentum prediction step.
    relaxation_scale : float
        The (positive) pressure-velocity coupling scale from
        :func:`momentum_relaxation_scale`.
    method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``.
    max_iterations : int
        Maximum number of linear solver iterations.
    tolerance : float
        Linear solver convergence tolerance.

    Returns
    -------
    ScalarField
        The solved pressure correction field, p_prime.
    """
    relaxation_scale = _validate_positive_number(relaxation_scale, "relaxation_scale")
    boundaries = [ZeroGradientBC(boundary) for boundary in ("left", "right", "top", "bottom")]
    return solve_pressure_correction(
        mesh,
        velocity_star,
        boundaries,
        dt=relaxation_scale,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# Step 3: velocity correction
# ---------------------------------------------------------------------------

def correct_velocity(
    mesh: Mesh,
    velocity_star: VectorField,
    pressure_correction_field: ScalarField,
    relaxation_scale: float,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
) -> VectorField:
    """Project velocity_star onto a divergence-free field using p_prime's gradient.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    velocity_star : VectorField
        The intermediate velocity field from the momentum prediction step.
    pressure_correction_field : ScalarField
        The pressure correction, p_prime, from
        :func:`solve_pressure_correction_step`.
    relaxation_scale : float
        The (positive) pressure-velocity coupling scale from
        :func:`momentum_relaxation_scale`.
    u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
        The Dirichlet velocity boundaries, re-applied after correction so
        the physical boundaries stay exact.

    Returns
    -------
    VectorField
        The corrected, divergence-free velocity field.
    """
    _validate_velocity_field(mesh, velocity_star, "velocity_star")
    _validate_pressure_field(mesh, pressure_correction_field)
    relaxation_scale = _validate_positive_number(relaxation_scale, "relaxation_scale")
    _validate_velocity_boundary_conditions(u_boundary_conditions, "u_boundary_conditions")
    _validate_velocity_boundary_conditions(v_boundary_conditions, "v_boundary_conditions")

    pressure_gradient = gradient(pressure_correction_field).values
    u_corrected = velocity_star.values[:, 0] - relaxation_scale * pressure_gradient[:, 0]
    v_corrected = velocity_star.values[:, 1] - relaxation_scale * pressure_gradient[:, 1]

    u_field = ScalarField(mesh, u_corrected)
    v_field = ScalarField(mesh, v_corrected)

    u_by_boundary = {bc.boundary: bc for bc in u_boundary_conditions}
    v_by_boundary = {bc.boundary: bc for bc in v_boundary_conditions}
    for boundary in _BOUNDARY_APPLICATION_ORDER:
        u_by_boundary[boundary].apply(u_field)
        v_by_boundary[boundary].apply(v_field)

    return VectorField(mesh, np.column_stack([u_field.values, v_field.values]))


# ---------------------------------------------------------------------------
# Step 4: residual computation
# ---------------------------------------------------------------------------

def compute_residuals(mesh: Mesh, previous_velocity: VectorField, corrected_velocity: VectorField) -> Dict[str, float]:
    """Compute the continuity and momentum-change residuals for one outer iteration.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    previous_velocity : VectorField
        The velocity field at the start of the outer iteration.
    corrected_velocity : VectorField
        The velocity field after pressure and velocity correction.

    Returns
    -------
    dict
        Root-mean-square residuals with keys ``"continuity"``,
        ``"u_momentum"`` and ``"v_momentum"``.
    """
    _validate_velocity_field(mesh, previous_velocity, "previous_velocity")
    _validate_velocity_field(mesh, corrected_velocity, "corrected_velocity")

    continuity_residual = float(np.sqrt(np.mean(divergence(corrected_velocity).values ** 2)))
    u_residual = float(
        np.sqrt(np.mean((corrected_velocity.values[:, 0] - previous_velocity.values[:, 0]) ** 2))
    )
    v_residual = float(
        np.sqrt(np.mean((corrected_velocity.values[:, 1] - previous_velocity.values[:, 1]) ** 2))
    )

    return {"continuity": continuity_residual, "u_momentum": u_residual, "v_momentum": v_residual}


# ---------------------------------------------------------------------------
# Step 5: convergence checking
# ---------------------------------------------------------------------------

def has_converged(residuals: Dict[str, float], tolerance: float) -> bool:
    """Return True once every residual component has dropped below tolerance.

    Parameters
    ----------
    residuals : dict
        A residual dictionary from :func:`compute_residuals`.
    tolerance : float
        The (positive) convergence tolerance.

    Returns
    -------
    bool
        Whether all residual components are below ``tolerance``.
    """
    if not isinstance(residuals, dict) or not residuals:
        raise TypeError("residuals must be a non-empty dict of residual values.")
    tolerance = _validate_positive_number(tolerance, "tolerance")
    return all(value < tolerance for value in residuals.values())


# ---------------------------------------------------------------------------
# Main SIMPLE loop
# ---------------------------------------------------------------------------

def solve_simple(
    mesh: Mesh,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    viscosity: float,
    initial_velocity: Optional[VectorField] = None,
    initial_pressure: Optional[ScalarField] = None,
    velocity_relaxation: float = 0.7,
    pressure_relaxation: float = 0.3,
    max_outer_iterations: int = 200,
    outer_tolerance: float = 1e-6,
    linear_method: str = "gauss_seidel",
    linear_max_iterations: int = 2000,
    linear_tolerance: float = 1e-8,
) -> SimpleSolverResult:
    """Run the SIMPLE algorithm to steady state on a structured 2D mesh.

    Each outer iteration performs momentum prediction, pressure correction,
    velocity correction, residual computation and a convergence check, in
    that order, until every residual drops below ``outer_tolerance`` or
    ``max_outer_iterations`` is reached.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D Cartesian mesh.
    u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances each, one per boundary
        ("left", "right", "top", "bottom").
    viscosity : float
        The (constant, positive) kinematic viscosity.
    initial_velocity : VectorField, optional
        Starting velocity guess. Defaults to zero everywhere.
    initial_pressure : ScalarField, optional
        Starting pressure guess. Defaults to zero everywhere.
    velocity_relaxation : float
        Under-relaxation factor applied to the momentum prediction, in
        (0, 1]. Defaults to ``0.7``.
    pressure_relaxation : float
        Under-relaxation factor applied to the pressure update, in (0, 1].
        Defaults to ``0.3``.
    max_outer_iterations : int
        Maximum number of SIMPLE outer iterations.
    outer_tolerance : float
        Outer iteration stops once every residual in
        :func:`compute_residuals` drops below this value.
    linear_method : str
        Either ``"jacobi"`` or ``"gauss_seidel"``, used for every inner
        linear solve (momentum prediction and pressure correction).
    linear_max_iterations : int
        Maximum number of iterations for each inner linear solve.
    linear_tolerance : float
        Convergence tolerance for each inner linear solve.

    Returns
    -------
    SimpleSolverResult
        The final velocity and pressure fields, plus convergence bookkeeping.
    """
    _validate_mesh_for_simple(mesh)
    _validate_velocity_boundary_conditions(u_boundary_conditions, "u_boundary_conditions")
    _validate_velocity_boundary_conditions(v_boundary_conditions, "v_boundary_conditions")
    viscosity = _validate_positive_number(viscosity, "viscosity")
    velocity_relaxation = _validate_relaxation_factor(velocity_relaxation, "velocity_relaxation")
    pressure_relaxation = _validate_relaxation_factor(pressure_relaxation, "pressure_relaxation")
    max_outer_iterations = _validate_positive_integer(max_outer_iterations, "max_outer_iterations")
    outer_tolerance = _validate_positive_number(outer_tolerance, "outer_tolerance")

    velocity = _validate_or_default_velocity(mesh, initial_velocity)
    pressure = _validate_or_default_pressure(mesh, initial_pressure)

    residual_history: List[Dict[str, float]] = []
    converged = False
    iterations_run = 0

    for iteration in range(max_outer_iterations):
        velocity_star, u_system, v_system = solve_momentum_predictor(
            mesh,
            velocity,
            u_boundary_conditions,
            v_boundary_conditions,
            viscosity,
            pressure,
            method=linear_method,
            max_iterations=linear_max_iterations,
            tolerance=linear_tolerance,
        )

        # Under-relax the momentum prediction before it feeds pressure
        # correction, to keep the outer loop from oscillating.
        relaxed_u = (
            velocity_relaxation * velocity_star.values[:, 0]
            + (1.0 - velocity_relaxation) * velocity.values[:, 0]
        )
        relaxed_v = (
            velocity_relaxation * velocity_star.values[:, 1]
            + (1.0 - velocity_relaxation) * velocity.values[:, 1]
        )
        velocity_star = VectorField(mesh, np.column_stack([relaxed_u, relaxed_v]))

        relaxation_scale = momentum_relaxation_scale(
            mesh, u_system, v_system, u_boundary_conditions, v_boundary_conditions
        )

        pressure_correction_field = solve_pressure_correction_step(
            mesh,
            velocity_star,
            relaxation_scale,
            method=linear_method,
            max_iterations=linear_max_iterations,
            tolerance=linear_tolerance,
        )

        new_velocity = correct_velocity(
            mesh,
            velocity_star,
            pressure_correction_field,
            relaxation_scale,
            u_boundary_conditions,
            v_boundary_conditions,
        )
        new_pressure = ScalarField(
            mesh, pressure.values + pressure_relaxation * pressure_correction_field.values
        )

        residuals = compute_residuals(mesh, velocity, new_velocity)
        residual_history.append(residuals)
        iterations_run = iteration + 1

        velocity = new_velocity
        pressure = new_pressure

        if has_converged(residuals, outer_tolerance):
            converged = True
            break

    return SimpleSolverResult(velocity, pressure, iterations_run, converged, residual_history)
