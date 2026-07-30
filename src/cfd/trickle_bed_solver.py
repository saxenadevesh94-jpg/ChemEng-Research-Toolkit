"""Educational transient trickle-bed Eulerian two-fluid solver.

Sprint 35 composes a trickle-bed reactor solver out of building blocks that
already exist in this package, rather than reimplementing flow physics that
earlier sprints already solved:

* the transient two-fluid momentum assembly and outer (Gauss-Seidel
  phase-coupling) iteration pattern from
  :mod:`~src.cfd.transient_eulerian_solver`
  (:func:`~src.cfd.transient_eulerian_solver.assemble_transient_phase_momentum_system`,
  :class:`~src.cfd.transient_eulerian_solver.TransientEulerianTwoFluidResult`);
* the Schiller-Naumann interphase drag model and per-phase momentum
  assembly from :mod:`~src.cfd.eulerian_solver`
  (:func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`,
  :func:`~src.cfd.eulerian_solver.interphase_drag_coefficient`);
* the Ergun-equation packed-bed resistance model from
  :mod:`~src.cfd.packed_bed`
  (:class:`~src.cfd.packed_bed.ErgunResistanceModel`,
  :class:`~src.cfd.packed_bed.PackedBedProperties`);
* the capillary pressure closures from :mod:`~src.cfd.capillary_pressure`
  (:class:`~src.cfd.capillary_pressure.CapillaryPressureModel`,
  :func:`~src.cfd.capillary_pressure.capillary_pressure_from_phase`);
* the force accounting framework from :mod:`~src.cfd.force_accounting`
  (:class:`~src.cfd.force_accounting.ForceAccounting`);
* the shared mixture pressure-correction solve from
  :meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.solve_mixture_flow`
  (itself the existing SIMPLE/Navier-Stokes solver);
* the pulse injection mechanism (a callable boundary-condition schedule)
  and pulse tracking framework from :mod:`~src.cfd.pulse_tracking`
  (:class:`~src.cfd.pulse_tracking.LiquidHoldupCalculator`,
  :class:`~src.cfd.pulse_tracking.PulseTracker`).

No relative-permeability closure exists anywhere else in this package, so
(matching the precedent set by :mod:`~src.cfd.eulerian_solver`, which added
the Schiller-Naumann drag correlation it needed, and
:mod:`~src.cfd.capillary_pressure`, which added its own closures)
:class:`RelativePermeabilityModel` and its concrete subclasses are
implemented here, following the same template-method pattern already used
by :class:`~src.cfd.capillary_pressure.CapillaryPressureModel`.

Physical coupling
------------------
A trickle bed percolates a continuous gas phase and a dispersed liquid
phase (droplets/rivulets, matching the interphase-drag correlation's usual
regime) through a fixed packing. Both phases keep their own momentum
equation, coupled through:

* interphase drag (Schiller-Naumann, reused unchanged);
* Ergun-equation packed-bed resistance, *divided by each phase's own
  relative permeability* (a function of liquid saturation): a phase whose
  relative permeability has dropped because the other phase is occupying
  more of the pore space experiences proportionally more resistance;
* a shared mixture pressure field (from the reused SIMPLE pressure
  correction), with capillary pressure applied as an additional pressure
  offset on the liquid phase's own momentum equation only
  (``p_liquid = p_mixture - p_c``, so ``p_gas = p_mixture`` acts as the
  reference): this reuses
  :func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`'s
  existing (per-call) pressure-gradient handling completely unchanged —
  only the pressure field it is given differs between the two phases.

As in :mod:`~src.cfd.transient_eulerian_solver`, volume-fraction (hence
liquid saturation) transport is not yet implemented anywhere in this
package, so liquid saturation - and therefore the capillary pressure and
relative permeability fields derived from it - is constant for the whole
simulation. Both are evaluated once, up front, and reused at every time
step, exactly as the (also constant) volume-fraction history is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .boundary_conditions import BoundaryCondition, FixedValueBC
from .capillary_pressure import CapillaryPressureModel, capillary_pressure_from_phase
from .equation import Equation
from .eulerian_solver import assemble_phase_momentum_system, interphase_drag_coefficient, relative_velocity_magnitude
from .field import ScalarField, VectorField
from .force_accounting import ForceAccounting
from .linear_system import LinearSystem
from .mesh import Mesh
from .multiphase import EulerianMultiphaseSystem, Phase
from .operators import gradient
from .packed_bed import ErgunResistanceModel, PackedBedProperties
from .pulse_tracking import LiquidHoldupCalculator, MonitoringLocations, PulseTracker
from .solvers import GaussSeidelSolver, JacobiSolver
from .transient_eulerian_solver import BoundaryConditionSchedule, TransientEulerianTwoFluidResult

_REQUIRED_BOUNDARIES = {"left", "right", "top", "bottom"}
_COMPONENT_INDEX = {"u": 0, "v": 1}

# Flooring conventions matching the ones already used elsewhere in this
# package (eulerian_solver's _MIN_VOLUME_FRACTION, capillary_pressure's
# _MIN_SATURATION): a relative permeability of exactly zero would divide by
# zero when scaling the Ergun resistance term.
_MIN_VOLUME_FRACTION = 1e-6
_MIN_RELATIVE_PERMEABILITY = 1e-6


# ---------------------------------------------------------------------------
# Descriptive bookkeeping
# ---------------------------------------------------------------------------

def trickle_bed_momentum_equation(phase_name: str, other_phase_name: str) -> Equation:
    """Return an Equation describing one phase's trickle-bed momentum equation.

    This is purely descriptive bookkeeping - it documents the PDE
    :func:`solve_trickle_bed` actually assembles and solves at every time
    step, using the same symbolic Equation container the rest of the CFD
    package uses.

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
        -gradient(p_<phase_name>) / rho_<phase_name> +
        viscosity_<phase_name> * laplacian(U_<phase_name>) +
        drag(U_<other_phase_name> - U_<phase_name>) / (alpha_<phase_name> * rho_<phase_name>) -
        ergun_resistance(U_<phase_name>) / (kr_<phase_name> * alpha_<phase_name> * rho_<phase_name>)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    if not isinstance(other_phase_name, str) or not other_phase_name.strip():
        raise ValueError("other_phase_name must be a non-empty string.")
    return Equation(
        lhs=f"ddt(U_{phase_name}) + convection(U_{phase_name})",
        rhs=(
            f"-gradient(p_{phase_name}) / rho_{phase_name} "
            f"+ viscosity_{phase_name} * laplacian(U_{phase_name}) "
            f"+ drag(U_{other_phase_name} - U_{phase_name}) / (alpha_{phase_name} * rho_{phase_name}) "
            f"- ergun_resistance(U_{phase_name}) / (kr_{phase_name} * alpha_{phase_name} * rho_{phase_name})"
        ),
    )


def relative_permeability_closure_equation(phase_name: str) -> Equation:
    """Return an Equation describing a phase's relative permeability closure.

    Parameters
    ----------
    phase_name : str
        Name of the phase this closure describes.

    Returns
    -------
    Equation
        ``kr_<phase_name> = f(S_<phase_name>)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    return Equation(lhs=f"kr_{phase_name}", rhs=f"f(S_{phase_name})")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_mesh_type(mesh: Mesh) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be an instance of Mesh, got {type(mesh).__name__}.")


def _validate_structured_mesh(mesh: Mesh) -> None:
    """Check that a mesh is a full, uniform, structured 2D grid."""
    _validate_mesh_type(mesh)
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("solve_trickle_bed only supports 2D Cartesian meshes.")

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


def _validate_saturation_field(field: object, mesh: Mesh, name: str) -> ScalarField:
    if not isinstance(field, ScalarField):
        raise TypeError(f"{name} must be a ScalarField, got {type(field).__name__}.")
    if field.mesh is not mesh:
        raise ValueError(f"{name} must be attached to the same mesh being solved.")
    if np.any(field.values < -1e-9) or np.any(field.values > 1.0 + 1e-9):
        raise ValueError(f"{name} values must lie within [0, 1].")
    return field


def _validate_relative_permeability_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be within the range [0, 1].")
    return number


def _validate_velocity_boundary_conditions(boundary_conditions: Sequence[BoundaryCondition], name: str) -> None:
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
    if callable(schedule):
        return
    _validate_velocity_boundary_conditions(schedule, name)


def _resolve_boundary_conditions(
    schedule: BoundaryConditionSchedule, time: float, name: str
) -> List[BoundaryCondition]:
    if callable(schedule):
        resolved = schedule(time)
        _validate_velocity_boundary_conditions(resolved, name)
        return list(resolved)
    return list(schedule)


def _boundary_cell_indices(mesh: Mesh, boundary: str) -> np.ndarray:
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
    raw_steps = total_time / dt
    n_steps = round(raw_steps)
    if n_steps < 1:
        raise ValueError("total_time must be large enough for at least one time step of size dt.")
    if not np.isclose(raw_steps, n_steps, rtol=1e-6, atol=1e-6):
        raise ValueError("total_time must be an integer multiple of dt.")
    return int(n_steps)


def _relax(previous: VectorField, updated: VectorField, relaxation: float) -> VectorField:
    blended = relaxation * updated.values + (1.0 - relaxation) * previous.values
    return VectorField(previous.mesh, blended)


def _rms_change(previous: VectorField, updated: VectorField) -> float:
    return float(np.sqrt(np.mean((updated.values - previous.values) ** 2)))


def _mixture_velocity(
    dispersed: Phase, continuous: Phase, dispersed_velocity: VectorField, continuous_velocity: VectorField
) -> VectorField:
    weight_dispersed = dispersed.volume_fraction.values * dispersed.density
    weight_continuous = continuous.volume_fraction.values * continuous.density
    numerator = (
        weight_dispersed[:, None] * dispersed_velocity.values
        + weight_continuous[:, None] * continuous_velocity.values
    )
    denominator = weight_dispersed + weight_continuous
    return VectorField(dispersed.mesh, numerator / denominator[:, None])


# ---------------------------------------------------------------------------
# Relative permeability models
# ---------------------------------------------------------------------------

class RelativePermeabilityModel(ABC):
    """Base class for liquid-saturation-dependent relative permeability closures.

    Subclasses implement :meth:`_evaluate`; this base class validates the
    shared input once (mesh attachment, value range) so every subclass gets
    identical, correct input checking for free, matching the template
    method pattern already used by
    :class:`~src.cfd.capillary_pressure.CapillaryPressureModel`.

    Parameters
    ----------
    mesh : Mesh
        The mesh every field this model consumes or returns is attached to.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Mesh.
    """

    def __init__(self, mesh: Mesh) -> None:
        _validate_mesh_type(mesh)
        self.mesh = mesh

    def relative_permeabilities(self, liquid_saturation: ScalarField) -> Tuple[ScalarField, ScalarField]:
        """Return ``(kr_liquid, kr_gas)`` for the given liquid saturation.

        Parameters
        ----------
        liquid_saturation : ScalarField
            Liquid saturation (fraction of void space occupied by liquid),
            attached to ``self.mesh``, with values in ``[0, 1]``.

        Returns
        -------
        tuple of ScalarField
            The liquid and gas relative permeability fields, each with
            values in ``[0, 1]``.

        Raises
        ------
        TypeError
            If ``liquid_saturation`` is not a ScalarField.
        ValueError
            If ``liquid_saturation`` is not attached to ``self.mesh`` or has
            values outside ``[0, 1]``.
        """
        liquid_saturation = _validate_saturation_field(liquid_saturation, self.mesh, "liquid_saturation")
        return self._evaluate(liquid_saturation)

    @abstractmethod
    def _evaluate(self, liquid_saturation: ScalarField) -> Tuple[ScalarField, ScalarField]:
        """Compute the closure formula on an already-validated saturation field."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mesh=<{self.mesh.n_cells} cells>)"


class CoreyRelativePermeabilityModel(RelativePermeabilityModel):
    """Corey-type power-law relative permeability closure.

        kr_liquid = S_liquid ** liquid_exponent
        kr_gas    = (1 - S_liquid) ** gas_exponent

    A standard, widely used closure shape for two-phase flow through
    porous/packed media: each phase's relative permeability falls to zero
    as it vanishes from the pore space and rises to one as it alone fills
    it.

    Parameters
    ----------
    mesh : Mesh
        The mesh every field this model returns is attached to.
    liquid_exponent, gas_exponent : float
        The (positive) Corey exponents for the liquid and gas phases.
        Defaults to ``2.0`` for both (a common quadratic choice in
        trickle-bed hydrodynamic models).

    Raises
    ------
    TypeError
        If ``mesh`` is not a Mesh, or an exponent is not a number.
    ValueError
        If an exponent is not positive.
    """

    def __init__(self, mesh: Mesh, liquid_exponent: float = 2.0, gas_exponent: float = 2.0) -> None:
        super().__init__(mesh)
        self.liquid_exponent = _validate_positive_number(liquid_exponent, "liquid_exponent")
        self.gas_exponent = _validate_positive_number(gas_exponent, "gas_exponent")

    def _evaluate(self, liquid_saturation: ScalarField) -> Tuple[ScalarField, ScalarField]:
        saturation = liquid_saturation.values
        kr_liquid = saturation ** self.liquid_exponent
        kr_gas = (1.0 - saturation) ** self.gas_exponent
        return ScalarField(self.mesh, kr_liquid), ScalarField(self.mesh, kr_gas)

    def __repr__(self) -> str:
        return (
            f"CoreyRelativePermeabilityModel(liquid_exponent={self.liquid_exponent}, "
            f"gas_exponent={self.gas_exponent})"
        )


class ConstantRelativePermeabilityModel(RelativePermeabilityModel):
    """Saturation-independent relative permeability closure.

    Returns the same constant relative permeability for each phase
    everywhere, regardless of liquid saturation. Useful as a test double or
    as a placeholder (``1.0`` for both phases reproduces the unmodified
    Ergun resistance).

    Parameters
    ----------
    mesh : Mesh
        The mesh every field this model returns is attached to.
    liquid_relative_permeability, gas_relative_permeability : float
        The constant relative permeability returned for each phase,
        everywhere. Must be within ``[0, 1]``. Both default to ``1.0``.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Mesh, or a value is not a number.
    ValueError
        If a value is outside ``[0, 1]``.
    """

    def __init__(
        self,
        mesh: Mesh,
        liquid_relative_permeability: float = 1.0,
        gas_relative_permeability: float = 1.0,
    ) -> None:
        super().__init__(mesh)
        self.liquid_relative_permeability = _validate_relative_permeability_value(
            liquid_relative_permeability, "liquid_relative_permeability"
        )
        self.gas_relative_permeability = _validate_relative_permeability_value(
            gas_relative_permeability, "gas_relative_permeability"
        )

    def _evaluate(self, liquid_saturation: ScalarField) -> Tuple[ScalarField, ScalarField]:
        kr_liquid = np.full(self.mesh.n_cells, self.liquid_relative_permeability)
        kr_gas = np.full(self.mesh.n_cells, self.gas_relative_permeability)
        return ScalarField(self.mesh, kr_liquid), ScalarField(self.mesh, kr_gas)

    def __repr__(self) -> str:
        return (
            f"ConstantRelativePermeabilityModel(liquid_relative_permeability="
            f"{self.liquid_relative_permeability}, gas_relative_permeability="
            f"{self.gas_relative_permeability})"
        )


# ---------------------------------------------------------------------------
# Trickle-bed per-phase momentum assembly
# ---------------------------------------------------------------------------

def assemble_trickle_bed_phase_momentum_system(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    bed_properties: PackedBedProperties,
    relative_permeability: ScalarField,
    boundary_conditions: Sequence[BoundaryCondition],
    component: str,
) -> LinearSystem:
    """Assemble one phase's two-fluid momentum system with relative-permeability-scaled Ergun resistance.

    Reuses :func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`
    for convection, diffusion, the (phase-specific) pressure gradient and
    interphase drag, then adds Ergun-equation packed-bed resistance
    (:class:`~src.cfd.packed_bed.ErgunResistanceModel`) divided by this
    phase's relative permeability as a further implicit sink, matching how
    :func:`~src.cfd.packed_bed.assemble_packed_bed_phase_momentum_system`
    adds its (unscaled) resistance term:

        (a_P + K/(alpha*rho) + (A + B*|U_P|)/(kr*alpha*rho)) * U_P + ... =
            ... + K/(alpha*rho) * U_other - grad(p)/rho

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    phase : Phase
        The phase percolating through the packed bed.
    velocity : VectorField
        The current velocity guess for ``phase``.
    other_phase_velocity : VectorField
        The current velocity guess for the other phase.
    drag_coefficient : ScalarField
        The interphase momentum exchange coefficient.
    pressure : ScalarField
        The pressure field driving this phase's momentum equation (may
        differ between phases, e.g. to add a capillary pressure offset).
    bed_properties : PackedBedProperties
        The packed bed's particle/fluid/packing properties.
    relative_permeability : ScalarField
        This phase's relative permeability field, attached to ``mesh``,
        with values in ``[0, 1]``.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances for this velocity component.
    component : str
        Either ``"u"`` or ``"v"``.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for this component.
    """
    if not isinstance(bed_properties, PackedBedProperties):
        raise TypeError(f"bed_properties must be a PackedBedProperties, got {type(bed_properties).__name__}.")
    if component not in _COMPONENT_INDEX:
        raise ValueError("component must be either 'u' or 'v'.")
    _validate_structured_mesh(mesh)
    _validate_scalar_field(relative_permeability, mesh, "relative_permeability")

    system = assemble_phase_momentum_system(
        mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure, boundary_conditions, component
    )

    model = ErgunResistanceModel(mesh, bed_properties)
    base_resistance = model.resistance_coefficient(velocity)
    kr = np.maximum(relative_permeability.values, _MIN_RELATIVE_PERMEABILITY)
    resistance = base_resistance.values / kr
    alpha = np.maximum(phase.volume_fraction.values, _MIN_VOLUME_FRACTION)
    resistance_scale = resistance / (alpha * phase.density)

    interior = _interior_mask(mesh, boundary_conditions)
    for cell_index in np.where(interior)[0]:
        cell_index = int(cell_index)
        current_diagonal = system.matrix.get(cell_index, cell_index)
        system.matrix.set(cell_index, cell_index, current_diagonal + resistance_scale[cell_index])

    return system


def assemble_transient_trickle_bed_phase_momentum_system(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    previous_velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    bed_properties: PackedBedProperties,
    relative_permeability: ScalarField,
    boundary_conditions: Sequence[BoundaryCondition],
    component: str,
    dt: float,
) -> LinearSystem:
    """Assemble one time step's implicit trickle-bed momentum system for a phase.

    Adds a first-order implicit (Backward Euler) time derivative on top of
    :func:`assemble_trickle_bed_phase_momentum_system`, exactly as
    :func:`~src.cfd.transient_eulerian_solver.assemble_transient_phase_momentum_system`
    does for the plain (non-trickle-bed) two-fluid momentum system.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    phase : Phase
        The phase this momentum equation is being assembled for.
    velocity : VectorField
        The current (within-time-step) velocity guess for ``phase``.
    previous_velocity : VectorField
        The velocity of ``phase`` at the start of this time step, U^n.
    other_phase_velocity : VectorField
        The current velocity guess for the other phase.
    drag_coefficient : ScalarField
        The interphase momentum exchange coefficient.
    pressure : ScalarField
        The pressure field driving this phase's momentum equation at the
        new time level.
    bed_properties : PackedBedProperties
        The packed bed's particle/fluid/packing properties.
    relative_permeability : ScalarField
        This phase's relative permeability field.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances for this velocity component.
    component : str
        Either ``"u"`` or ``"v"``.
    dt : float
        The (positive) time step size.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for this component.
    """
    index = _validate_component(component)
    _validate_structured_mesh(mesh)
    _validate_velocity_field(previous_velocity, "previous_velocity")
    dt = _validate_positive_number(dt, "dt")

    system = assemble_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure,
        bed_properties, relative_permeability, boundary_conditions, component,
    )

    interior = _interior_mask(mesh, boundary_conditions)
    inv_dt = 1.0 / dt
    for cell_index in np.where(interior)[0]:
        cell_index = int(cell_index)
        current_diagonal = system.matrix.get(cell_index, cell_index)
        system.matrix.set(cell_index, cell_index, current_diagonal + inv_dt)

    system.rhs[interior] += inv_dt * previous_velocity.values[interior, index]

    return system


def _solve_transient_trickle_bed_phase_velocity(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    previous_velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    bed_properties: PackedBedProperties,
    relative_permeability: ScalarField,
    u_boundary_conditions: Sequence[BoundaryCondition],
    v_boundary_conditions: Sequence[BoundaryCondition],
    dt: float,
    method: str,
    max_iterations: int,
    tolerance: float,
) -> VectorField:
    """Solve both velocity components of one phase's transient trickle-bed momentum equation."""
    u_system = assemble_transient_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_phase_velocity, drag_coefficient, pressure,
        bed_properties, relative_permeability, u_boundary_conditions, "u", dt,
    )
    v_system = assemble_transient_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_phase_velocity, drag_coefficient, pressure,
        bed_properties, relative_permeability, v_boundary_conditions, "v", dt,
    )

    u_solver = _build_solver(method, max_iterations, tolerance)
    v_solver = _build_solver(method, max_iterations, tolerance)

    u_values = u_solver.solve(u_system, initial_guess=velocity.values[:, 0])
    v_values = v_solver.solve(v_system, initial_guess=velocity.values[:, 1])

    return VectorField(mesh, np.column_stack([u_values, v_values]))


def _advance_trickle_bed_time_step(
    mesh: Mesh,
    liquid: Phase,
    gas: Phase,
    liquid_name: str,
    gas_name: str,
    liquid_velocity_prev: VectorField,
    gas_velocity_prev: VectorField,
    liquid_pressure: ScalarField,
    gas_pressure: ScalarField,
    bed_properties: PackedBedProperties,
    liquid_relative_permeability: ScalarField,
    gas_relative_permeability: ScalarField,
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
) -> Tuple[VectorField, VectorField, ScalarField, int, bool, List[Dict[str, float]]]:
    """Advance both phases' velocities by one implicit trickle-bed time step.

    The liquid phase is treated as the dispersed phase for the interphase
    drag correlation (droplets/rivulets within a continuous gas phase,
    matching typical trickle-flow-regime treatment); the gas phase is
    treated as continuous.
    """
    liquid_velocity = VectorField(mesh, liquid_velocity_prev.values.copy())
    gas_velocity = VectorField(mesh, gas_velocity_prev.values.copy())
    drag_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    residual_history: List[Dict[str, float]] = []
    converged = False
    iterations_run = 0

    for iteration in range(max_outer_iterations):
        relative_speed = relative_velocity_magnitude(liquid_velocity, gas_velocity)
        drag_field = interphase_drag_coefficient(
            mesh, gas.density, gas.viscosity, liquid.volume_fraction, relative_speed, particle_diameter,
        )

        new_liquid_velocity = _solve_transient_trickle_bed_phase_velocity(
            mesh, liquid, liquid_velocity, liquid_velocity_prev, gas_velocity, drag_field, liquid_pressure,
            bed_properties, liquid_relative_permeability, u_boundary_conditions, v_boundary_conditions, dt,
            linear_method, linear_max_iterations, linear_tolerance,
        )
        new_gas_velocity = _solve_transient_trickle_bed_phase_velocity(
            mesh, gas, gas_velocity, gas_velocity_prev, liquid_velocity, drag_field, gas_pressure,
            bed_properties, gas_relative_permeability, u_boundary_conditions, v_boundary_conditions, dt,
            linear_method, linear_max_iterations, linear_tolerance,
        )

        relaxed_liquid_velocity = _relax(liquid_velocity, new_liquid_velocity, velocity_relaxation)
        relaxed_gas_velocity = _relax(gas_velocity, new_gas_velocity, velocity_relaxation)

        residuals = {
            f"{liquid_name}_velocity": _rms_change(liquid_velocity, relaxed_liquid_velocity),
            f"{gas_name}_velocity": _rms_change(gas_velocity, relaxed_gas_velocity),
        }
        residual_history.append(residuals)
        iterations_run = iteration + 1

        liquid_velocity = relaxed_liquid_velocity
        gas_velocity = relaxed_gas_velocity

        if all(value < outer_tolerance for value in residuals.values()):
            converged = True
            break

    return liquid_velocity, gas_velocity, drag_field, iterations_run, converged, residual_history


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class TrickleBedResult(TransientEulerianTwoFluidResult):
    """Store the full time history of a transient trickle-bed two-fluid solve.

    A subclass of
    :class:`~src.cfd.transient_eulerian_solver.TransientEulerianTwoFluidResult`
    so that :class:`~src.cfd.pulse_tracking.LiquidHoldupCalculator` and
    :class:`~src.cfd.pulse_tracking.PulseTracker` - which both require a
    ``TransientEulerianTwoFluidResult`` instance - work against it
    unmodified. See that class for the inherited ``time_points``,
    ``velocity_history``, ``pressure_history``, ``volume_fraction_history``,
    ``drag_coefficient_history`` and per-step convergence bookkeeping.

    Parameters
    ----------
    trickle_bed_properties : PackedBedProperties
        The packed bed's particle/fluid/packing properties used for this
        solve.
    capillary_pressure_field : ScalarField
        The (constant, since saturation is not transported) capillary
        pressure field used to offset the liquid phase's pressure.
    liquid_relative_permeability_field, gas_relative_permeability_field : ScalarField
        The (constant) relative permeability fields used to scale each
        phase's Ergun resistance.
    liquid_phase_name, gas_phase_name : str
        Names of the liquid and gas phases within this result.

    Attributes
    ----------
    holdup_calculator : LiquidHoldupCalculator
        Built automatically from this result and ``liquid_phase_name``.
    pulse_tracker : PulseTracker or None
        ``None`` until :meth:`attach_pulse_tracker` is called (or
        :func:`solve_trickle_bed` is given ``monitoring_locations``).
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
        trickle_bed_properties: PackedBedProperties,
        capillary_pressure_field: ScalarField,
        liquid_relative_permeability_field: ScalarField,
        gas_relative_permeability_field: ScalarField,
        liquid_phase_name: str,
        gas_phase_name: str,
    ) -> None:
        super().__init__(
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
            bed_properties=None,
        )
        self.trickle_bed_properties = trickle_bed_properties
        self.capillary_pressure_field = capillary_pressure_field
        self.liquid_relative_permeability_field = liquid_relative_permeability_field
        self.gas_relative_permeability_field = gas_relative_permeability_field
        self.liquid_phase_name = liquid_phase_name
        self.gas_phase_name = gas_phase_name
        self.holdup_calculator = LiquidHoldupCalculator(self, liquid_phase_name)
        self.pulse_tracker: Optional[PulseTracker] = None

    def relative_permeability(self, phase_name: str) -> ScalarField:
        """Return the (constant) relative permeability field for a phase.

        Raises
        ------
        KeyError
            If no phase with that name is in this result.
        """
        if phase_name == self.liquid_phase_name:
            return self.liquid_relative_permeability_field
        if phase_name == self.gas_phase_name:
            return self.gas_relative_permeability_field
        raise KeyError(f"no phase named '{phase_name}' in this result.")

    def attach_pulse_tracker(
        self,
        monitoring_locations: MonitoringLocations,
        signal: str = "velocity_magnitude",
        arrival_fraction: float = 0.1,
    ) -> PulseTracker:
        """Build and store a :class:`~src.cfd.pulse_tracking.PulseTracker` for this result."""
        self.pulse_tracker = PulseTracker(
            self, self.liquid_phase_name, monitoring_locations, signal=signal, arrival_fraction=arrival_fraction
        )
        return self.pulse_tracker

    def build_force_accounting(self, phase_name: str, step: int = -1) -> ForceAccounting:
        """Build a ForceAccounting report for one phase at one stored time step.

        Extends
        :meth:`~src.cfd.transient_eulerian_solver.TransientEulerianTwoFluidResult.build_force_accounting`
        (which registers ``"pressure_gradient"`` and ``"interphase_drag"``)
        with two more contributions:

        * ``"packed_bed_resistance"`` - the Ergun resistance force, divided
          by this phase's relative permeability, reusing
          :class:`~src.cfd.packed_bed.ErgunResistanceModel`;
        * ``"capillary_pressure"`` - registered only for the liquid phase,
          the extra force implied by its pressure offset
          (``p_liquid = p_mixture - p_c``).

        Raises
        ------
        KeyError
            If no phase with that name was solved.
        ValueError
            If this result does not contain exactly two phases.
        IndexError
            If ``step`` is out of range.
        """
        accounting = super().build_force_accounting(phase_name, step)

        this_velocity = self.velocity(phase_name, step)
        this_volume_fraction = self.volume_fraction(phase_name, step)
        mesh = this_velocity.mesh
        kr_field = self.relative_permeability(phase_name)

        model = ErgunResistanceModel(mesh, self.trickle_bed_properties)
        base_resistance = model.resistance_coefficient(this_velocity)
        kr = np.maximum(kr_field.values, _MIN_RELATIVE_PERMEABILITY)
        resistance = base_resistance.values / kr
        source = -resistance[:, None] * this_velocity.values
        accounting.register("packed_bed_resistance", VectorField(mesh, source))

        if phase_name == self.liquid_phase_name:
            capillary_force = this_volume_fraction.values[:, None] * gradient(self.capillary_pressure_field).values
            accounting.register("capillary_pressure", VectorField(mesh, capillary_force))

        return accounting

    def __repr__(self) -> str:
        return (
            f"TrickleBedResult(liquid_phase={self.liquid_phase_name!r}, gas_phase={self.gas_phase_name!r}, "
            f"n_steps={self.n_steps}, dt={self.dt}, total_time={self.total_time}, converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# Main trickle-bed solve
# ---------------------------------------------------------------------------

def solve_trickle_bed(
    system: EulerianMultiphaseSystem,
    liquid_phase: str,
    particle_diameter: float,
    bed_properties: PackedBedProperties,
    capillary_pressure_model: CapillaryPressureModel,
    relative_permeability_model: RelativePermeabilityModel,
    surface_tension: float,
    contact_angle: float,
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
    monitoring_locations: Optional[MonitoringLocations] = None,
    pulse_signal: str = "velocity_magnitude",
    pulse_arrival_fraction: float = 0.1,
) -> TrickleBedResult:
    """March a trickle-bed reactor's coupled gas/liquid momentum equations forward in time.

    At every time step:

        1. the shared mixture pressure is re-solved from the
           volume-fraction-weighted mixture momentum equation
           (:meth:`~src.cfd.multiphase.EulerianMultiphaseSystem.solve_mixture_flow`),
           warm-started from the previous step's mixture velocity and
           pressure;
        2. the liquid phase's own pressure is taken as the mixture pressure
           minus the (constant) capillary pressure field, and the gas
           phase's own pressure is the mixture pressure unchanged;
        3. both phases' momentum equations - each including interphase
           drag and a relative-permeability-scaled Ergun resistance term -
           are advanced by one first-order implicit (Backward Euler) time
           step (:func:`assemble_transient_trickle_bed_phase_momentum_system`),
           iterated Gauss-Seidel style with under-relaxation until both
           phases' velocity residuals drop below ``outer_tolerance`` or
           ``max_outer_iterations`` is reached;
        4. the resulting velocity, pressure and (currently unchanging)
           volume-fraction fields are appended to the result's history.

    Parameters
    ----------
    system : EulerianMultiphaseSystem
        A system containing exactly two phases, attached to a structured,
        uniformly-spaced 2D Cartesian mesh.
    liquid_phase : str
        Name of the liquid phase within ``system``. The other phase is
        treated as the gas phase. The liquid phase is treated as dispersed
        for the interphase-drag correlation; the gas phase as continuous.
    particle_diameter : float
        The (positive) liquid droplet/rivulet diameter used by the
        interphase drag correlation. Distinct from ``bed_properties``'s own
        (packing) particle diameter.
    bed_properties : PackedBedProperties
        The packed bed's particle/fluid/packing properties.
    capillary_pressure_model : CapillaryPressureModel
        The capillary pressure closure to evaluate from liquid saturation.
    relative_permeability_model : RelativePermeabilityModel
        The relative permeability closure to evaluate from liquid
        saturation.
    surface_tension : float
        Gas-liquid interfacial surface tension, forwarded to
        ``capillary_pressure_model``. Must be non-negative.
    contact_angle : float
        Liquid-solid contact angle in radians, forwarded to
        ``capillary_pressure_model``. Must be within ``[0, pi]``.
    dt : float
        The (positive) time step size.
    total_time : float
        The (positive) total simulation time. Must be an integer multiple
        of ``dt``.
    u_boundary_conditions, v_boundary_conditions : BoundaryConditionSchedule
        Either a fixed list/tuple of exactly four FixedValueBC instances,
        or a callable ``time -> list of four FixedValueBC`` re-evaluated at
        every time step (e.g. a pulsed liquid inlet).
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
    monitoring_locations : mapping of str to coordinates, or sequence of coordinates, optional
        If given, forwarded to
        :meth:`TrickleBedResult.attach_pulse_tracker` once the solve
        finishes, building ``result.pulse_tracker`` automatically.
    pulse_signal : str
        Forwarded to :class:`~src.cfd.pulse_tracking.PulseTracker` if
        ``monitoring_locations`` is given.
    pulse_arrival_fraction : float
        Forwarded to :class:`~src.cfd.pulse_tracking.PulseTracker` if
        ``monitoring_locations`` is given.

    Returns
    -------
    TrickleBedResult
        The full time history of both phases' velocities, the shared
        pressure, liquid holdup (via ``result.holdup_calculator``), force
        accounting (via ``result.build_force_accounting``), and pulse
        tracking (via ``result.pulse_tracker``, if ``monitoring_locations``
        was given).
    """
    if not isinstance(system, EulerianMultiphaseSystem):
        raise TypeError(f"system must be an EulerianMultiphaseSystem, got {type(system).__name__}.")
    if system.n_phases != 2:
        raise ValueError("solve_trickle_bed only supports two-phase (gas-liquid) systems.")

    mesh = system.mesh
    _validate_structured_mesh(mesh)

    if not isinstance(liquid_phase, str):
        raise TypeError(f"liquid_phase must be a string, got {type(liquid_phase).__name__}.")
    if liquid_phase not in system.phase_names:
        raise ValueError(f"liquid_phase '{liquid_phase}' is not a phase in this system.")
    gas_phase = next(name for name in system.phase_names if name != liquid_phase)

    particle_diameter = _validate_positive_number(particle_diameter, "particle_diameter")
    if not isinstance(bed_properties, PackedBedProperties):
        raise TypeError(f"bed_properties must be a PackedBedProperties, got {type(bed_properties).__name__}.")
    if not isinstance(capillary_pressure_model, CapillaryPressureModel):
        raise TypeError(
            "capillary_pressure_model must be a CapillaryPressureModel, "
            f"got {type(capillary_pressure_model).__name__}."
        )
    if not isinstance(relative_permeability_model, RelativePermeabilityModel):
        raise TypeError(
            "relative_permeability_model must be a RelativePermeabilityModel, "
            f"got {type(relative_permeability_model).__name__}."
        )

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

    liquid = system.get_phase(liquid_phase)
    gas = system.get_phase(gas_phase)

    capillary_pressure_field = capillary_pressure_from_phase(
        capillary_pressure_model, liquid, bed_properties, surface_tension, contact_angle
    )
    liquid_kr_field, gas_kr_field = relative_permeability_model.relative_permeabilities(liquid.volume_fraction)

    liquid_velocity = VectorField(mesh, liquid.velocity.values.copy())
    gas_velocity = VectorField(mesh, gas.velocity.values.copy())
    pressure = ScalarField(mesh, liquid.pressure.values.copy())

    time_points: List[float] = [0.0]
    velocity_history: Dict[str, List[VectorField]] = {
        liquid_phase: [liquid_velocity],
        gas_phase: [gas_velocity],
    }
    pressure_history: List[ScalarField] = [pressure]
    volume_fraction_history: Dict[str, List[ScalarField]] = {
        liquid_phase: [ScalarField(mesh, liquid.volume_fraction.values.copy())],
        gas_phase: [ScalarField(mesh, gas.volume_fraction.values.copy())],
    }
    drag_coefficient_history: List[ScalarField] = [ScalarField(mesh, np.zeros(mesh.n_cells))]
    iterations_per_step: List[int] = [0]
    converged_per_step: List[bool] = [True]
    residual_history_per_step: List[List[Dict[str, float]]] = [[]]

    for step in range(1, n_steps + 1):
        time = step * dt
        u_bcs = _resolve_boundary_conditions(u_boundary_conditions, time, "u_boundary_conditions")
        v_bcs = _resolve_boundary_conditions(v_boundary_conditions, time, "v_boundary_conditions")

        warm_start_velocity = _mixture_velocity(liquid, gas, liquid_velocity, gas_velocity)
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
        liquid_pressure = ScalarField(mesh, pressure.values - capillary_pressure_field.values)

        (
            liquid_velocity,
            gas_velocity,
            drag_field,
            iterations_run,
            converged,
            residuals,
        ) = _advance_trickle_bed_time_step(
            mesh,
            liquid,
            gas,
            liquid_phase,
            gas_phase,
            liquid_velocity,
            gas_velocity,
            liquid_pressure,
            pressure,
            bed_properties,
            liquid_kr_field,
            gas_kr_field,
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
        )

        time_points.append(time)
        velocity_history[liquid_phase].append(liquid_velocity)
        velocity_history[gas_phase].append(gas_velocity)
        pressure_history.append(pressure)
        volume_fraction_history[liquid_phase].append(ScalarField(mesh, liquid.volume_fraction.values.copy()))
        volume_fraction_history[gas_phase].append(ScalarField(mesh, gas.volume_fraction.values.copy()))
        drag_coefficient_history.append(drag_field)
        iterations_per_step.append(iterations_run)
        converged_per_step.append(converged)
        residual_history_per_step.append(residuals)

    result = TrickleBedResult(
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
        trickle_bed_properties=bed_properties,
        capillary_pressure_field=capillary_pressure_field,
        liquid_relative_permeability_field=liquid_kr_field,
        gas_relative_permeability_field=gas_kr_field,
        liquid_phase_name=liquid_phase,
        gas_phase_name=gas_phase,
    )

    if monitoring_locations is not None:
        result.attach_pulse_tracker(
            monitoring_locations, signal=pulse_signal, arrival_fraction=pulse_arrival_fraction
        )

    return result
