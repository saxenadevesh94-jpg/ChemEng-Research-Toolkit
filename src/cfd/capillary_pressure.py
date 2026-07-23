"""Educational capillary pressure models for two-phase flow in packed beds.

Sprint 29 adds capillary pressure closures for the gas-liquid two-fluid model
built in :mod:`~src.cfd.eulerian_solver` when the continuous/dispersed
phases percolate through a packed bed (:mod:`~src.cfd.packed_bed`). Capillary
pressure is the pressure difference sustained across curved gas-liquid
interfaces trapped between packing particles:

    p_c = p_gas - p_liquid = f(S_l, eps, d_p, sigma, cos(theta))

where ``S_l`` is liquid saturation (the fraction of the void space occupied
by liquid), ``eps`` is bed porosity, ``d_p`` is particle diameter, ``sigma``
is the gas-liquid surface tension and ``theta`` is the liquid-solid contact
angle. All three models below share this same five-argument interface
through the :class:`CapillaryPressureModel` template method
(:meth:`CapillaryPressureModel.capillary_pressure`), which validates the
shared inputs once and delegates the actual closure formula to each
subclass:

* :class:`LeverettJFunctionModel` — the classic Leverett J-function closure,
  built on a Kozeny-Carman permeability estimate (the same porosity/particle
  -diameter grouping already used for Ergun's viscous coefficient in
  :mod:`~src.cfd.packed_bed`).
* :class:`AttouFerschneiderModel` — a simplified capillary-pressure-versus
  -saturation closure in the style used by trickle-bed reactor hydrodynamic
  models (e.g. Attou & Ferschneider, 1999): capillary pressure diverges as
  liquid saturation approaches zero and vanishes as saturation approaches
  one. It is a representative, illustrative closure rather than a verbatim
  reproduction of any single paper's fitted constants.
* :class:`ConstantCapillaryPressureModel` — a saturation-independent
  placeholder, useful for tests and for solvers that only need a fixed
  capillary pressure offset.

This module reuses the same Mesh, ScalarField and validation conventions
already used throughout the CFD package, plus the existing
:class:`~src.cfd.multiphase.Phase` and
:class:`~src.cfd.packed_bed.PackedBedProperties` containers via
:func:`capillary_pressure_from_phase` — it does not reimplement mesh
handling, field storage or bed-property bookkeeping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .equation import Equation
from .field import ScalarField
from .mesh import Mesh
from .multiphase import Phase
from .packed_bed import PackedBedProperties

# A liquid saturation of exactly zero would divide by zero in
# AttouFerschneiderModel's (1 - S_l) / S_l closure; flooring it keeps the
# model finite while still producing a very large (correctly signed)
# capillary pressure, matching the flooring convention already used for
# relative velocity and volume fraction elsewhere in this package.
_MIN_SATURATION = 1e-6

# Coefficients of Leverett's original empirical J-function correlation.
_LEVERETT_COEFFICIENTS = (1.417, -2.120, 1.263)


# ---------------------------------------------------------------------------
# Descriptive bookkeeping
# ---------------------------------------------------------------------------

def capillary_pressure_closure_equation(phase_name: str) -> Equation:
    """Return an Equation describing a phase's capillary pressure closure.

    This is purely descriptive bookkeeping — it documents the closure
    relationship a capillary pressure model actually evaluates, using the
    same symbolic Equation container the rest of the CFD package uses.

    Parameters
    ----------
    phase_name : str
        Name of the liquid phase this closure describes.

    Returns
    -------
    Equation
        ``p_c_<phase_name> = f(S_<phase_name>, porosity, particle_diameter,
        surface_tension, contact_angle)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    return Equation(
        lhs=f"p_c_{phase_name}",
        rhs=(
            f"f(S_{phase_name}, porosity, particle_diameter, "
            f"surface_tension, contact_angle)"
        ),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_mesh_type(mesh: object) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be an instance of Mesh, got {type(mesh).__name__}.")


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


def _validate_contact_angle(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"contact_angle must be a number, got {type(value).__name__}.")
    number = float(value)
    if not (0.0 <= number <= np.pi):
        raise ValueError("contact_angle must be within [0, pi] radians.")
    return number


def _validate_saturation_field(field: object, mesh: Mesh, name: str) -> ScalarField:
    if not isinstance(field, ScalarField):
        raise TypeError(f"{name} must be a ScalarField, got {type(field).__name__}.")
    if field.mesh is not mesh:
        raise ValueError(f"{name} must be attached to the same mesh being solved.")
    if np.any(field.values < -1e-9) or np.any(field.values > 1.0 + 1e-9):
        raise ValueError(f"{name} values must lie within [0, 1].")
    return field


# ---------------------------------------------------------------------------
# Capillary pressure model base class
# ---------------------------------------------------------------------------

class CapillaryPressureModel(ABC):
    """Base class for capillary-pressure-versus-saturation closures.

    Subclasses implement :meth:`_evaluate`; this base class validates the
    shared five-argument interface once (mesh attachment, value ranges) so
    every subclass gets identical, correct input checking for free.

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

    def capillary_pressure(
        self,
        liquid_saturation: ScalarField,
        porosity: float,
        particle_diameter: float,
        surface_tension: float,
        contact_angle: float,
    ) -> ScalarField:
        """Return the per-cell capillary pressure, p_c = p_gas - p_liquid.

        Parameters
        ----------
        liquid_saturation : ScalarField
            Liquid saturation (fraction of void space occupied by liquid),
            attached to ``self.mesh``, with values in ``[0, 1]``.
        porosity : float
            Bed voidage, strictly between 0 and 1.
        particle_diameter : float
            Diameter of the (assumed spherical, uniform) packing particles.
            Must be positive.
        surface_tension : float
            Gas-liquid interfacial surface tension. Must be non-negative.
        contact_angle : float
            Liquid-solid contact angle, in radians, within ``[0, pi]``.

        Returns
        -------
        ScalarField
            The capillary pressure at every cell.

        Raises
        ------
        TypeError
            If any input has the wrong type.
        ValueError
            If ``liquid_saturation`` is not attached to ``self.mesh`` or has
            values outside ``[0, 1]``, if ``porosity`` is not strictly
            between 0 and 1, if ``particle_diameter`` is not positive, if
            ``surface_tension`` is negative, or if ``contact_angle`` is
            outside ``[0, pi]``.
        """
        liquid_saturation = _validate_saturation_field(liquid_saturation, self.mesh, "liquid_saturation")
        porosity = _validate_porosity(porosity)
        particle_diameter = _validate_positive_number(particle_diameter, "particle_diameter")
        surface_tension = _validate_non_negative_number(surface_tension, "surface_tension")
        contact_angle = _validate_contact_angle(contact_angle)
        return self._evaluate(liquid_saturation, porosity, particle_diameter, surface_tension, contact_angle)

    @abstractmethod
    def _evaluate(
        self,
        liquid_saturation: ScalarField,
        porosity: float,
        particle_diameter: float,
        surface_tension: float,
        contact_angle: float,
    ) -> ScalarField:
        """Compute the closure formula on already-validated inputs."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mesh=<{self.mesh.n_cells} cells>)"


# ---------------------------------------------------------------------------
# Leverett J-function model
# ---------------------------------------------------------------------------

class LeverettJFunctionModel(CapillaryPressureModel):
    """Leverett J-function capillary pressure closure.

    Uses the Kozeny-Carman estimate of bed permeability (the same
    porosity/particle-diameter grouping underlying Ergun's viscous
    coefficient in :class:`~src.cfd.packed_bed.PackedBedProperties`):

        k = eps^3 * d_p^2 / (150 * (1 - eps)^2)

    and Leverett's original empirical correlation:

        J(S_l) = 1.417*(1 - S_l) - 2.120*(1 - S_l)^2 + 1.263*(1 - S_l)^3

        p_c = sigma * cos(theta) * sqrt(eps / k) * J(S_l)
    """

    def _evaluate(
        self,
        liquid_saturation: ScalarField,
        porosity: float,
        particle_diameter: float,
        surface_tension: float,
        contact_angle: float,
    ) -> ScalarField:
        permeability = (porosity ** 3 * particle_diameter ** 2) / (150.0 * (1.0 - porosity) ** 2)
        scale = surface_tension * np.cos(contact_angle) * np.sqrt(porosity / permeability)

        one_minus_s = 1.0 - liquid_saturation.values
        c1, c2, c3 = _LEVERETT_COEFFICIENTS
        j_function = c1 * one_minus_s + c2 * one_minus_s ** 2 + c3 * one_minus_s ** 3

        return ScalarField(self.mesh, scale * j_function)


# ---------------------------------------------------------------------------
# Attou-Ferschneider-style model
# ---------------------------------------------------------------------------

class AttouFerschneiderModel(CapillaryPressureModel):
    """Trickle-bed-style capillary pressure closure.

    A simplified capillary-pressure-versus-saturation relationship in the
    style used by trickle-bed reactor hydrodynamic models (e.g. Attou &
    Ferschneider, 1999): capillary pressure grows without bound as liquid
    saturation approaches zero (an increasingly liquid-starved bed sustains
    an increasingly large interfacial pressure jump) and vanishes as
    saturation approaches one (a fully liquid-filled bed has no gas-liquid
    interface left to sustain one):

        p_c = (sigma * cos(theta) / d_p) * sqrt((1 - eps) / eps)
              * (1 - S_l) / S_l

    ``S_l`` is floored at a small positive value before the division so the
    closure stays finite at (and only asymptotically approaches) zero
    saturation.
    """

    def _evaluate(
        self,
        liquid_saturation: ScalarField,
        porosity: float,
        particle_diameter: float,
        surface_tension: float,
        contact_angle: float,
    ) -> ScalarField:
        saturation = np.maximum(liquid_saturation.values, _MIN_SATURATION)
        structural_factor = np.sqrt((1.0 - porosity) / porosity)
        scale = surface_tension * np.cos(contact_angle) / particle_diameter

        pressure = scale * structural_factor * (1.0 - saturation) / saturation
        return ScalarField(self.mesh, pressure)


# ---------------------------------------------------------------------------
# Constant model
# ---------------------------------------------------------------------------

class ConstantCapillaryPressureModel(CapillaryPressureModel):
    """Saturation-independent capillary pressure closure.

    Returns the same constant capillary pressure everywhere, regardless of
    liquid saturation, porosity, particle diameter, surface tension or
    contact angle. Useful as a test double or as a placeholder for solvers
    that only need a fixed capillary pressure offset.

    Parameters
    ----------
    mesh : Mesh
        The mesh every field this model returns is attached to.
    capillary_pressure_value : float
        The constant capillary pressure value returned everywhere. Must be
        non-negative.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Mesh, or ``capillary_pressure_value`` is not a
        number.
    ValueError
        If ``capillary_pressure_value`` is negative.
    """

    def __init__(self, mesh: Mesh, capillary_pressure_value: float) -> None:
        super().__init__(mesh)
        self.capillary_pressure_value = _validate_non_negative_number(
            capillary_pressure_value, "capillary_pressure_value"
        )

    def _evaluate(
        self,
        liquid_saturation: ScalarField,
        porosity: float,
        particle_diameter: float,
        surface_tension: float,
        contact_angle: float,
    ) -> ScalarField:
        return ScalarField(self.mesh, np.full(self.mesh.n_cells, self.capillary_pressure_value))

    def __repr__(self) -> str:
        return f"ConstantCapillaryPressureModel(capillary_pressure_value={self.capillary_pressure_value})"


# ---------------------------------------------------------------------------
# Two-fluid / packed-bed integration
# ---------------------------------------------------------------------------

def capillary_pressure_from_phase(
    model: CapillaryPressureModel,
    liquid_phase: Phase,
    bed_properties: PackedBedProperties,
    surface_tension: float,
    contact_angle: float,
) -> ScalarField:
    """Evaluate a capillary pressure model directly from existing framework objects.

    Reuses the liquid phase's volume fraction, from the Eulerian two-fluid
    framework (:class:`~src.cfd.multiphase.Phase`), as the liquid
    saturation, and the packed bed's porosity and particle diameter, from
    :class:`~src.cfd.packed_bed.PackedBedProperties`, rather than requiring
    callers to re-extract those values by hand.

    Parameters
    ----------
    model : CapillaryPressureModel
        The capillary pressure closure to evaluate.
    liquid_phase : Phase
        The liquid phase; its ``volume_fraction`` field supplies liquid
        saturation.
    bed_properties : PackedBedProperties
        The packed bed's properties; its ``porosity`` and
        ``particle_diameter`` supply the corresponding closure arguments.
    surface_tension : float
        Gas-liquid interfacial surface tension. Must be non-negative.
    contact_angle : float
        Liquid-solid contact angle, in radians, within ``[0, pi]``.

    Returns
    -------
    ScalarField
        The capillary pressure at every cell, from
        :meth:`CapillaryPressureModel.capillary_pressure`.

    Raises
    ------
    TypeError
        If ``model`` is not a CapillaryPressureModel, ``liquid_phase`` is
        not a Phase, or ``bed_properties`` is not a PackedBedProperties.
    """
    if not isinstance(model, CapillaryPressureModel):
        raise TypeError(f"model must be a CapillaryPressureModel, got {type(model).__name__}.")
    if not isinstance(liquid_phase, Phase):
        raise TypeError(f"liquid_phase must be a Phase, got {type(liquid_phase).__name__}.")
    if not isinstance(bed_properties, PackedBedProperties):
        raise TypeError(f"bed_properties must be a PackedBedProperties, got {type(bed_properties).__name__}.")

    return model.capillary_pressure(
        liquid_phase.volume_fraction,
        bed_properties.porosity,
        bed_properties.particle_diameter,
        surface_tension,
        contact_angle,
    )
