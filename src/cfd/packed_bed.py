"""Educational Ergun-equation packed-bed resistance model for structured 2D CFD meshes.

Sprint 28 adds a momentum sink representing the resistance a packed bed of
stationary solid particles exerts on a fluid flowing through it, using the
classic Ergun equation:

    -dp/dz = 150 * mu * (1 - eps)^2 / (eps^3 * dp^2) * U
             + 1.75 * rho * (1 - eps) / (eps^3 * dp) * U^2

The first (viscous, laminar) term dominates at low Reynolds number, the
second (inertial, turbulent) term at high Reynolds number, matching the same
"laminar branch / high-Re branch" shape already used for interphase drag in
:mod:`~src.cfd.eulerian_solver` (Schiller-Naumann drag).

For a CFD momentum equation, the same relationship is recast as a
per-unit-volume momentum sink opposing the local velocity vector:

    S = -(A + B * |U|) * U,     A = 150*mu*(1-eps)^2/(eps^3*dp^2)
                                 B = 1.75*rho*(1-eps)/(eps^3*dp)

:class:`ErgunResistanceModel` computes ``A + B*|U|`` and ``S`` per cell.
:func:`assemble_packed_bed_phase_momentum_system` then adds that sink,
implicitly, on top of an already-assembled two-fluid phase momentum system
by calling :func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`
directly rather than reimplementing convection, diffusion, pressure-gradient
or drag assembly. The packing itself is not modelled as an Eulerian phase
(it is stationary and solid, so it has no momentum equation of its own) —
it only ever appears as a resistance term acting on whichever fluid phase
percolates through it.

This module reuses the same Mesh, ScalarField, VectorField, Equation,
LinearSystem, Phase and two-fluid solver infrastructure built in earlier
sprints rather than reimplementing any of it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .boundary_conditions import BoundaryCondition
from .equation import Equation
from .eulerian_solver import assemble_phase_momentum_system
from .field import ScalarField, VectorField
from .linear_system import LinearSystem
from .mesh import Mesh
from .multiphase import Phase

_COMPONENT_INDEX = {"u": 0, "v": 1}

# A dispersed-phase volume fraction of exactly zero would divide by zero when
# normalising the resistance term by alpha_k * rho_k; flooring it keeps the
# momentum system well-posed, matching the same flooring used for interphase
# drag in eulerian_solver.
_MIN_VOLUME_FRACTION = 1e-6


# ---------------------------------------------------------------------------
# Descriptive bookkeeping
# ---------------------------------------------------------------------------

def packed_bed_momentum_equation(phase_name: str) -> Equation:
    """Return an Equation describing a phase's momentum equation with packed-bed resistance.

    This is purely descriptive bookkeeping — it documents the PDE
    :func:`assemble_packed_bed_phase_momentum_system` actually assembles,
    using the same symbolic Equation container the rest of the CFD package
    uses.

    Parameters
    ----------
    phase_name : str
        Name of the phase this equation describes.

    Returns
    -------
    Equation
        ``convection(U_<phase_name>) = -gradient(p) / rho_<phase_name> +
        viscosity_<phase_name> * laplacian(U_<phase_name>) -
        ergun_resistance(U_<phase_name>) / (alpha_<phase_name> * rho_<phase_name>)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    return Equation(
        lhs=f"convection(U_{phase_name})",
        rhs=(
            f"-gradient(p) / rho_{phase_name} "
            f"+ viscosity_{phase_name} * laplacian(U_{phase_name}) "
            f"- ergun_resistance(U_{phase_name}) / (alpha_{phase_name} * rho_{phase_name})"
        ),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _validate_non_negative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def _validate_porosity(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"porosity must be a number, got {type(value).__name__}.")
    number = float(value)
    if not (0.0 < number < 1.0):
        raise ValueError("porosity must be strictly between 0 and 1.")
    return number


def _validate_mesh_type(mesh: Mesh) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be an instance of Mesh, got {type(mesh).__name__}.")


def _validate_velocity_field(field: object, mesh: Mesh, name: str) -> None:
    if not isinstance(field, VectorField):
        raise TypeError(f"{name} must be a VectorField, got {type(field).__name__}.")
    if field.mesh is not mesh:
        raise ValueError(f"{name} must be attached to the same mesh being solved.")
    if field.values.shape[1] != 2:
        raise ValueError(f"{name} must have exactly two components (u, v).")


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


# ---------------------------------------------------------------------------
# Packed bed properties
# ---------------------------------------------------------------------------

class PackedBedProperties:
    """Store the particle, fluid and packing properties of a packed bed.

    Parameters
    ----------
    particle_diameter : float
        Diameter of the (assumed spherical, uniform) packing particles.
        Must be positive.
    porosity : float
        Bed voidage (void volume fraction), strictly between 0 and 1.
    fluid_viscosity : float
        Dynamic viscosity of the fluid flowing through the bed. Must be
        positive.
    fluid_density : float
        Density of the fluid flowing through the bed. Must be positive.
    superficial_velocity : float, optional
        Default superficial (empty-tower) velocity used by
        :meth:`pressure_gradient` and :meth:`pressure_drop` when no velocity
        is passed explicitly. Must be non-negative if given.

    Raises
    ------
    TypeError
        If any input has the wrong type.
    ValueError
        If ``particle_diameter``, ``fluid_viscosity`` or ``fluid_density``
        are not positive, if ``porosity`` is not strictly between 0 and 1,
        or if ``superficial_velocity`` is given and negative.
    """

    def __init__(
        self,
        particle_diameter: float,
        porosity: float,
        fluid_viscosity: float,
        fluid_density: float,
        superficial_velocity: Optional[float] = None,
    ) -> None:
        self.particle_diameter = _validate_positive_number(particle_diameter, "particle_diameter")
        self.porosity = _validate_porosity(porosity)
        self.fluid_viscosity = _validate_positive_number(fluid_viscosity, "fluid_viscosity")
        self.fluid_density = _validate_positive_number(fluid_density, "fluid_density")
        self.superficial_velocity = (
            None
            if superficial_velocity is None
            else _validate_non_negative_number(superficial_velocity, "superficial_velocity")
        )

    @property
    def viscous_coefficient(self) -> float:
        """Ergun's laminar (viscous) resistance coefficient, A."""
        eps = self.porosity
        return 150.0 * self.fluid_viscosity * (1.0 - eps) ** 2 / (eps ** 3 * self.particle_diameter ** 2)

    @property
    def inertial_coefficient(self) -> float:
        """Ergun's turbulent (inertial) resistance coefficient, B."""
        eps = self.porosity
        return 1.75 * self.fluid_density * (1.0 - eps) / (eps ** 3 * self.particle_diameter)

    def _resolve_velocity(self, velocity: Optional[float]) -> float:
        if velocity is None:
            if self.superficial_velocity is None:
                raise ValueError(
                    "velocity must be given because no superficial_velocity was set at construction."
                )
            return self.superficial_velocity
        return _validate_non_negative_number(velocity, "velocity")

    def pressure_gradient(self, velocity: Optional[float] = None) -> float:
        """Return the Ergun-equation pressure gradient magnitude, -dp/dz.

        Parameters
        ----------
        velocity : float, optional
            Superficial velocity to evaluate at. Defaults to
            ``self.superficial_velocity`` if not given.

        Returns
        -------
        float
            ``A * U + B * U**2``, using the coefficients from
            :attr:`viscous_coefficient` and :attr:`inertial_coefficient`.
        """
        speed = self._resolve_velocity(velocity)
        return self.viscous_coefficient * speed + self.inertial_coefficient * speed ** 2

    def pressure_drop(self, bed_length: float, velocity: Optional[float] = None) -> float:
        """Return the total pressure drop across a bed of the given length.

        Parameters
        ----------
        bed_length : float
            Length of the bed in the flow direction. Must be positive.
        velocity : float, optional
            Superficial velocity to evaluate at. Defaults to
            ``self.superficial_velocity`` if not given.

        Returns
        -------
        float
            ``pressure_gradient(velocity) * bed_length``.
        """
        bed_length = _validate_positive_number(bed_length, "bed_length")
        return self.pressure_gradient(velocity) * bed_length

    def __repr__(self) -> str:
        return (
            f"PackedBedProperties(particle_diameter={self.particle_diameter}, "
            f"porosity={self.porosity}, fluid_viscosity={self.fluid_viscosity}, "
            f"fluid_density={self.fluid_density})"
        )


# ---------------------------------------------------------------------------
# Ergun resistance model
# ---------------------------------------------------------------------------

class ErgunResistanceModel:
    """Compute Ergun-equation packed-bed momentum resistance fields on a mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh every field this model returns is attached to.
    properties : PackedBedProperties
        The bed's particle/fluid/packing properties.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Mesh or ``properties`` is not a
        PackedBedProperties.
    """

    def __init__(self, mesh: Mesh, properties: PackedBedProperties) -> None:
        _validate_mesh_type(mesh)
        if not isinstance(properties, PackedBedProperties):
            raise TypeError(f"properties must be a PackedBedProperties, got {type(properties).__name__}.")
        self.mesh = mesh
        self.properties = properties

    def resistance_coefficient(self, velocity: VectorField) -> ScalarField:
        """Return the per-cell Ergun resistance coefficient, A + B*|U|.

        This is the scalar multiplying velocity in the momentum sink term
        (see :meth:`momentum_source`), the packed-bed analogue of the
        interphase drag coefficient K returned by
        :func:`~src.cfd.eulerian_solver.interphase_drag_coefficient`.

        Parameters
        ----------
        velocity : VectorField
            The local fluid velocity, attached to ``self.mesh``.

        Returns
        -------
        ScalarField
            ``A + B * |U|`` at every cell.
        """
        _validate_velocity_field(velocity, self.mesh, "velocity")
        speed = np.sqrt(np.sum(velocity.values ** 2, axis=1))
        coefficient = self.properties.viscous_coefficient + self.properties.inertial_coefficient * speed
        return ScalarField(self.mesh, coefficient)

    def momentum_source(self, velocity: VectorField) -> VectorField:
        """Return the per-cell Ergun resistance source, -(A + B*|U|) * U.

        This is a momentum sink, opposing the local flow direction, with
        units of force per unit volume.

        Parameters
        ----------
        velocity : VectorField
            The local fluid velocity, attached to ``self.mesh``.

        Returns
        -------
        VectorField
            ``-(A + B * |U|) * U`` at every cell.
        """
        coefficient = self.resistance_coefficient(velocity)
        source = -coefficient.values[:, None] * velocity.values
        return VectorField(self.mesh, source)

    def __repr__(self) -> str:
        return f"ErgunResistanceModel(properties={self.properties!r})"


# ---------------------------------------------------------------------------
# Two-fluid momentum system integration
# ---------------------------------------------------------------------------

def assemble_packed_bed_phase_momentum_system(
    mesh: Mesh,
    phase: Phase,
    velocity: VectorField,
    other_phase_velocity: VectorField,
    drag_coefficient: ScalarField,
    pressure: ScalarField,
    bed_properties: PackedBedProperties,
    boundary_conditions: Sequence[BoundaryCondition],
    component: str,
) -> LinearSystem:
    """Assemble one phase's two-fluid momentum system with packed-bed resistance added.

    Reuses :func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`
    for convection, diffusion, the shared pressure gradient and interphase
    drag, then adds the Ergun-equation packed-bed resistance as a further
    implicit sink on top, normalised by the same ``alpha_phase * rho_phase``
    factor already used for drag so both source terms live on the same
    (density-divided) footing:

        (a_P + K/(alpha*rho) + (A + B*|U_P|)/(alpha*rho)) * U_P + ... =
            ... + K/(alpha*rho) * U_other - grad(p)/rho

    The resistance term is added only to interior rows; boundary rows
    (already pinned to their FixedValueBC value by
    :func:`~src.cfd.eulerian_solver.assemble_phase_momentum_system`) are
    left untouched.

    Parameters
    ----------
    mesh : Mesh
        A structured, uniform 2D mesh.
    phase : Phase
        The phase percolating through the packed bed. Its ``density`` and
        ``volume_fraction`` supply the resistance term's normalisation.
    velocity : VectorField
        The current velocity guess for ``phase``, used both by
        ``assemble_phase_momentum_system`` and to evaluate the
        Ergun resistance coefficient.
    other_phase_velocity : VectorField
        The current velocity guess for the other phase, forwarded to
        ``assemble_phase_momentum_system`` for the drag term.
    drag_coefficient : ScalarField
        The interphase momentum exchange coefficient, forwarded to
        ``assemble_phase_momentum_system``.
    pressure : ScalarField
        The shared pressure field.
    bed_properties : PackedBedProperties
        The packed bed's particle/fluid/packing properties.
    boundary_conditions : Sequence[BoundaryCondition]
        Exactly four FixedValueBC instances for this velocity component.
    component : str
        Either ``"u"`` or ``"v"``.

    Returns
    -------
    LinearSystem
        The assembled matrix and right-hand-side vector for this component,
        including the packed-bed resistance term.
    """
    if not isinstance(bed_properties, PackedBedProperties):
        raise TypeError(f"bed_properties must be a PackedBedProperties, got {type(bed_properties).__name__}.")
    if component not in _COMPONENT_INDEX:
        raise ValueError("component must be either 'u' or 'v'.")

    system = assemble_phase_momentum_system(
        mesh, phase, velocity, other_phase_velocity, drag_coefficient, pressure, boundary_conditions, component
    )

    model = ErgunResistanceModel(mesh, bed_properties)
    resistance = model.resistance_coefficient(velocity)
    alpha = np.maximum(phase.volume_fraction.values, _MIN_VOLUME_FRACTION)
    resistance_scale = resistance.values / (alpha * phase.density)

    interior = _interior_mask(mesh, boundary_conditions)
    for cell_index in np.where(interior)[0]:
        cell_index = int(cell_index)
        current_diagonal = system.matrix.get(cell_index, cell_index)
        system.matrix.set(cell_index, cell_index, current_diagonal + resistance_scale[cell_index])

    return system
