"""Educational Eulerian multiphase framework for structured 2D CFD meshes.

This module introduces the data structures needed to describe an Eulerian
multiphase flow — several interpenetrating phases (e.g. gas and liquid),
each with its own density, viscosity, volume fraction, velocity field and
pressure field, all sharing one mesh.

It reuses the same Mesh, ScalarField, VectorField, BoundaryCondition,
Equation and LinearSystem infrastructure built in earlier sprints, and
delegates any actual flow solving to the existing
:func:`~src.cfd.navier_stokes.solve_navier_stokes` (SIMPLE-based) solver via
the volume-fraction-weighted mixture-model simplification: one shared
momentum equation is solved using mixture density and viscosity rather than
one momentum equation per phase. Nothing here reimplements mesh handling,
field storage, linear system assembly or the SIMPLE/Navier-Stokes solve.

Per-phase momentum coupling (drag, interfacial transfer) and volume-fraction
transport are left for later sprints — this sprint is the framework that
those will build on.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .boundary_conditions import BoundaryCondition
from .equation import Equation
from .field import ScalarField, VectorField
from .mesh import Mesh
from .navier_stokes import NavierStokesResult, solve_navier_stokes

_VOLUME_FRACTION_TOLERANCE = 1e-6

FieldLike = Union[ScalarField, float, int, np.ndarray, Sequence[float]]


def phase_continuity_equation(phase_name: str) -> Equation:
    """Return an Equation describing a phase's volume-fraction transport.

    This is purely descriptive bookkeeping — it documents the PDE an
    Eulerian multiphase solver would need to advance in time, using the
    same symbolic Equation container the rest of the CFD package uses,
    rather than encoding it as a bare string. No transport is solved here.

    Parameters
    ----------
    phase_name : str
        Name of the phase the equation describes.

    Returns
    -------
    Equation
        ``ddt(alpha_<phase_name>) = -divergence(alpha_<phase_name> * U)``
    """
    if not isinstance(phase_name, str) or not phase_name.strip():
        raise ValueError("phase_name must be a non-empty string.")
    return Equation(
        lhs=f"ddt(alpha_{phase_name})",
        rhs=f"-divergence(alpha_{phase_name} * U)",
    )


def volume_fraction_closure_equation(phase_names: Sequence[str]) -> Equation:
    """Return an Equation describing the volume-fraction closure constraint.

    Parameters
    ----------
    phase_names : Sequence[str]
        Names of every phase in the system.

    Returns
    -------
    Equation
        ``alpha_<p1> + alpha_<p2> + ... = 1``
    """
    if not isinstance(phase_names, (list, tuple)) or not phase_names:
        raise ValueError("phase_names must be a non-empty list or tuple of strings.")
    lhs = " + ".join(f"alpha_{name}" for name in phase_names)
    return Equation(lhs=lhs, rhs="1")


def _validate_mesh(mesh: object) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be an instance of Mesh, got {type(mesh).__name__}.")


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


class Phase:
    """Store the material properties and fields of a single Eulerian phase.

    Parameters
    ----------
    name : str
        Unique, non-empty name identifying the phase (e.g. ``"gas"``).
    mesh : Mesh
        The mesh this phase's fields are attached to.
    density : float
        The (constant, positive) phase density.
    viscosity : float
        The (constant, positive) phase viscosity.
    volume_fraction : ScalarField, float, or array-like
        The phase's volume fraction. A single number is broadcast to every
        cell; an array-like or ScalarField must have one value per cell, in
        ``[0, 1]``.
    velocity : VectorField, optional
        The phase's velocity field. Defaults to zero everywhere, with one
        component per mesh dimension.
    pressure : ScalarField, optional
        The phase's pressure field. Defaults to zero everywhere.

    Raises
    ------
    TypeError
        If any input has the wrong type.
    ValueError
        If ``name`` is empty, ``density``/``viscosity`` are not positive,
        ``volume_fraction`` values fall outside ``[0, 1]``, or a field
        argument is attached to a different mesh.
    """

    def __init__(
        self,
        name: str,
        mesh: Mesh,
        density: float,
        viscosity: float,
        volume_fraction: FieldLike,
        velocity: Optional[VectorField] = None,
        pressure: Optional[ScalarField] = None,
    ) -> None:
        self.name = self._validate_name(name)
        _validate_mesh(mesh)
        self.mesh = mesh
        self.density = _validate_positive_number(density, "density")
        self.viscosity = _validate_positive_number(viscosity, "viscosity")
        self.volume_fraction = self._build_volume_fraction_field(mesh, volume_fraction)
        self.velocity = self._build_velocity_field(mesh, velocity)
        self.pressure = self._build_pressure_field(mesh, pressure)

    @staticmethod
    def _validate_name(name: object) -> str:
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}.")
        stripped = name.strip()
        if not stripped:
            raise ValueError("name must not be empty or whitespace-only.")
        return stripped

    @staticmethod
    def _build_volume_fraction_field(mesh: Mesh, volume_fraction: FieldLike) -> ScalarField:
        if isinstance(volume_fraction, ScalarField):
            if volume_fraction.mesh is not mesh:
                raise ValueError("volume_fraction must be attached to the same mesh as the phase.")
            field = ScalarField(mesh, volume_fraction.values.copy())
        elif isinstance(volume_fraction, bool):
            raise TypeError("volume_fraction must be a ScalarField, a number, or an array-like, got bool.")
        elif isinstance(volume_fraction, (int, float)):
            field = ScalarField(mesh, np.full(mesh.n_cells, float(volume_fraction)))
        elif isinstance(volume_fraction, (np.ndarray, list, tuple)):
            field = ScalarField(mesh, np.asarray(volume_fraction, dtype=float))
        else:
            raise TypeError(
                "volume_fraction must be a ScalarField, a number, or an array-like, "
                f"got {type(volume_fraction).__name__}."
            )

        if np.any(field.values < -1e-9) or np.any(field.values > 1.0 + 1e-9):
            raise ValueError("volume_fraction values must lie within [0, 1].")
        return field

    @staticmethod
    def _build_velocity_field(mesh: Mesh, velocity: Optional[VectorField]) -> VectorField:
        if velocity is None:
            n_components = mesh.cell_centers.shape[1]
            return VectorField(mesh, np.zeros((mesh.n_cells, n_components)))
        if not isinstance(velocity, VectorField):
            raise TypeError(f"velocity must be a VectorField, got {type(velocity).__name__}.")
        if velocity.mesh is not mesh:
            raise ValueError("velocity must be attached to the same mesh as the phase.")
        return VectorField(mesh, velocity.values.copy())

    @staticmethod
    def _build_pressure_field(mesh: Mesh, pressure: Optional[ScalarField]) -> ScalarField:
        if pressure is None:
            return ScalarField(mesh, np.zeros(mesh.n_cells))
        if not isinstance(pressure, ScalarField):
            raise TypeError(f"pressure must be a ScalarField, got {type(pressure).__name__}.")
        if pressure.mesh is not mesh:
            raise ValueError("pressure must be attached to the same mesh as the phase.")
        return ScalarField(mesh, pressure.values.copy())

    def __repr__(self) -> str:
        return (
            f"Phase(name={self.name!r}, density={self.density}, viscosity={self.viscosity}, "
            f"mean_volume_fraction={self.volume_fraction.mean():.4f})"
        )


class EulerianMultiphaseSystem:
    """Store a set of Eulerian phases sharing one mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh every phase in the system is attached to.
    phases : Sequence[Phase]
        At least two Phase instances, each with a unique name, all attached
        to ``mesh``. Their volume fractions must sum to (approximately) one
        in every cell.
    volume_fraction_tolerance : float
        The (positive) absolute tolerance used when checking that volume
        fractions sum to unity. Defaults to ``1e-6``.

    Raises
    ------
    TypeError
        If ``mesh``, ``phases``, or any entry of ``phases`` has the wrong
        type.
    ValueError
        If fewer than two phases are given, phase names are not unique, a
        phase is attached to a different mesh, or volume fractions do not
        sum to one in every cell.
    """

    def __init__(
        self,
        mesh: Mesh,
        phases: Sequence[Phase],
        volume_fraction_tolerance: float = _VOLUME_FRACTION_TOLERANCE,
    ) -> None:
        _validate_mesh(mesh)
        self.mesh = mesh
        self.volume_fraction_tolerance = _validate_positive_number(
            volume_fraction_tolerance, "volume_fraction_tolerance"
        )
        self.phases = self._validate_phases(mesh, phases)
        self._phase_by_name = {phase.name: phase for phase in self.phases}
        self._validate_volume_fraction_sum()

    @staticmethod
    def _validate_phases(mesh: Mesh, phases: Sequence[Phase]) -> List[Phase]:
        if not isinstance(phases, (list, tuple)):
            raise TypeError(f"phases must be a list or tuple of Phase instances, got {type(phases).__name__}.")
        if len(phases) < 2:
            raise ValueError("an Eulerian multiphase system requires at least two phases.")

        names = set()
        for phase in phases:
            if not isinstance(phase, Phase):
                raise TypeError(f"every entry in phases must be a Phase instance, got {type(phase).__name__}.")
            if phase.mesh is not mesh:
                raise ValueError(f"phase '{phase.name}' is not attached to the system mesh.")
            if phase.name in names:
                raise ValueError(f"duplicate phase name '{phase.name}'.")
            names.add(phase.name)
        return list(phases)

    def _validate_volume_fraction_sum(self) -> None:
        total = self.volume_fraction_sum().values
        if not np.allclose(total, 1.0, atol=self.volume_fraction_tolerance):
            worst = float(np.max(np.abs(total - 1.0)))
            raise ValueError(
                "phase volume fractions must sum to 1 in every cell "
                f"(largest deviation: {worst:.3e}, tolerance: {self.volume_fraction_tolerance:.3e})."
            )

    @property
    def n_phases(self) -> int:
        """Number of phases in the system."""
        return len(self.phases)

    @property
    def phase_names(self) -> List[str]:
        """Names of every phase in the system, in storage order."""
        return [phase.name for phase in self.phases]

    def get_phase(self, name: str) -> Phase:
        """Return the phase with the given name.

        Raises
        ------
        TypeError
            If ``name`` is not a string.
        KeyError
            If no phase with that name exists.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}.")
        try:
            return self._phase_by_name[name]
        except KeyError as exc:
            raise KeyError(f"no phase named '{name}' in this system.") from exc

    def volume_fraction_sum(self) -> ScalarField:
        """Return the per-cell sum of every phase's volume fraction."""
        total = np.zeros(self.mesh.n_cells)
        for phase in self.phases:
            total += phase.volume_fraction.values
        return ScalarField(self.mesh, total)

    def revalidate_volume_fractions(self) -> None:
        """Re-check that volume fractions still sum to unity in every cell.

        ``Phase.volume_fraction`` is a mutable field, so callers may update
        it in place between solves (e.g. after advecting it elsewhere); this
        re-runs the same check performed at construction time.

        Raises
        ------
        ValueError
            If the volume fractions no longer sum to one in every cell.
        """
        self._validate_volume_fraction_sum()

    def mixture_density_field(self) -> ScalarField:
        """Return the per-cell, volume-fraction-weighted mixture density."""
        total = np.zeros(self.mesh.n_cells)
        for phase in self.phases:
            total += phase.volume_fraction.values * phase.density
        return ScalarField(self.mesh, total)

    def mixture_viscosity_field(self) -> ScalarField:
        """Return the per-cell, volume-fraction-weighted mixture viscosity."""
        total = np.zeros(self.mesh.n_cells)
        for phase in self.phases:
            total += phase.volume_fraction.values * phase.viscosity
        return ScalarField(self.mesh, total)

    def mixture_density(self) -> float:
        """Return the domain-averaged mixture density.

        Existing solvers in this package (e.g.
        :func:`~src.cfd.navier_stokes.solve_navier_stokes`) assume a single,
        constant fluid density, so this reduces the per-cell mixture density
        field to one representative value for use with them.
        """
        return float(self.mixture_density_field().mean())

    def mixture_viscosity(self) -> float:
        """Return the domain-averaged mixture viscosity.

        See :meth:`mixture_density` for why this is a single scalar rather
        than a field.
        """
        return float(self.mixture_viscosity_field().mean())

    def mixture_velocity(self) -> VectorField:
        """Return the per-cell, mass-weighted mixture velocity.

        This is the standard Eulerian mixture-model average:
        ``sum(alpha_i * rho_i * U_i) / sum(alpha_i * rho_i)``. The
        denominator is always strictly positive because volume fractions
        sum to one and every phase density is positive.
        """
        n_components = self.mesh.cell_centers.shape[1]
        numerator = np.zeros((self.mesh.n_cells, n_components))
        denominator = np.zeros(self.mesh.n_cells)
        for phase in self.phases:
            weight = phase.volume_fraction.values * phase.density
            numerator += weight[:, None] * phase.velocity.values
            denominator += weight
        return VectorField(self.mesh, numerator / denominator[:, None])

    def solve_mixture_flow(
        self,
        u_boundary_conditions: Sequence[BoundaryCondition],
        v_boundary_conditions: Sequence[BoundaryCondition],
        **kwargs: object,
    ) -> NavierStokesResult:
        """Solve for a shared mixture velocity/pressure field.

        This applies the Eulerian mixture-model simplification: instead of
        one momentum equation per phase, a single momentum equation is
        solved for the whole mixture using the volume-fraction-weighted
        mixture density and viscosity (:meth:`mixture_density`,
        :meth:`mixture_viscosity`) as the fluid properties. The solve itself
        is entirely delegated to
        :func:`~src.cfd.navier_stokes.solve_navier_stokes` — no momentum,
        pressure-correction or SIMPLE-loop logic is reimplemented here.

        Parameters
        ----------
        u_boundary_conditions, v_boundary_conditions : Sequence[BoundaryCondition]
            Forwarded to :func:`~src.cfd.navier_stokes.solve_navier_stokes`.
        **kwargs
            Any other keyword argument accepted by
            :func:`~src.cfd.navier_stokes.solve_navier_stokes` (e.g.
            ``max_outer_iterations``), forwarded unchanged.

        Returns
        -------
        NavierStokesResult
            The mixture velocity and pressure fields, plus convergence
            bookkeeping.
        """
        return solve_navier_stokes(
            self.mesh,
            u_boundary_conditions,
            v_boundary_conditions,
            density=self.mixture_density(),
            dynamic_viscosity=self.mixture_viscosity(),
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"EulerianMultiphaseSystem(n_phases={self.n_phases}, phases={self.phase_names!r})"
