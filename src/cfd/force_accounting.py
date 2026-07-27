"""Force accounting framework for Eulerian multiphase CFD momentum sources.

Sprint 31 adds a modular bookkeeping layer for the per-cell momentum-force
contributions that already flow through this package's Eulerian multiphase
solvers — gravity, interphase drag
(:func:`~src.cfd.eulerian_solver.interphase_drag_coefficient`), packed-bed
resistance (:class:`~src.cfd.packed_bed.ErgunResistanceModel`), capillary
pressure (:mod:`~src.cfd.capillary_pressure`), mechanical dispersion, or any
other user-defined force — so they can be named, stored, retrieved, summed
and reported on consistently.

This module does not compute any new physics itself: every force is a plain
:class:`~src.cfd.field.VectorField` (typically a force-per-unit-volume
momentum source, matching the convention already used by
:meth:`~src.cfd.packed_bed.ErgunResistanceModel.momentum_source`) supplied by
the caller, wherever it came from. :class:`ForceAccounting` only registers,
validates, aggregates and reports on those fields, reusing the existing
Mesh, ScalarField and VectorField infrastructure rather than reimplementing
mesh handling or field storage.

Keeping accounting separate from computation keeps this module usable for
later transient and trickle-bed force analysis without depending on, or
constraining, how any particular force is modelled.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .field import ScalarField, VectorField
from .mesh import Mesh

# A denominator of exactly zero would divide by zero when normalising force
# contributions in cells where every registered force happens to vanish;
# flooring it keeps normalized_contributions finite (returning zero shares
# there) rather than raising or producing NaNs.
_MIN_MAGNITUDE = 1e-300


def _validate_mesh_type(mesh: object) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be an instance of Mesh, got {type(mesh).__name__}.")


def _validate_name(value: object, field_name: str = "name") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty or whitespace-only.")
    return stripped


def _validate_metadata(metadata: Optional[dict]) -> dict:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}.")
    return dict(metadata)


# ---------------------------------------------------------------------------
# ForceContribution
# ---------------------------------------------------------------------------

class ForceContribution:
    """Store one named momentum-force contribution as a mesh-backed vector field.

    Parameters
    ----------
    name : str
        Unique, non-empty name identifying the force (e.g. ``"gravity"``,
        ``"interphase_drag"``).
    force : VectorField
        The force (conventionally force-per-unit-volume) field itself.
    units : str
        Human-readable units for ``force`` (e.g. ``"N/m^3"``). Must be a
        non-empty string.
    metadata : dict, optional
        Arbitrary additional bookkeeping (e.g. source model name, phase, or
        physical parameters). Stored as a shallow copy so later mutation of
        the caller's dict does not affect this contribution. Defaults to an
        empty dict.

    Raises
    ------
    TypeError
        If ``force`` is not a VectorField, or ``name``/``units`` is not a
        string, or ``metadata`` is given and is not a dict.
    ValueError
        If ``name`` or ``units`` is empty or whitespace-only.
    """

    def __init__(
        self,
        name: str,
        force: VectorField,
        units: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self.name = _validate_name(name)
        if not isinstance(force, VectorField):
            raise TypeError(f"force must be a VectorField, got {type(force).__name__}.")
        self.force = force
        self.units = _validate_name(units, "units")
        self.metadata = _validate_metadata(metadata)

    @property
    def mesh(self) -> Mesh:
        """The mesh this contribution's force field is attached to."""
        return self.force.mesh

    def magnitude(self) -> ScalarField:
        """Return the per-cell magnitude (Euclidean norm) of this force."""
        magnitude = np.sqrt(np.sum(self.force.values ** 2, axis=1))
        return ScalarField(self.force.mesh, magnitude)

    def __repr__(self) -> str:
        return (
            f"ForceContribution(name={self.name!r}, units={self.units!r}, "
            f"mesh=<{self.force.mesh.n_cells} cells>)"
        )


# ---------------------------------------------------------------------------
# ForceAccounting
# ---------------------------------------------------------------------------

class ForceAccounting:
    """Register, store and report per-cell momentum-force contributions on one mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh every registered force contribution must be attached to.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Mesh.
    """

    def __init__(self, mesh: Mesh) -> None:
        _validate_mesh_type(mesh)
        self.mesh = mesh
        self._contributions: Dict[str, ForceContribution] = {}

    # -- registration ------------------------------------------------------

    def register(
        self,
        name: str,
        force: VectorField,
        units: str = "N/m^3",
        metadata: Optional[dict] = None,
    ) -> ForceContribution:
        """Build a :class:`ForceContribution` and register it under ``name``.

        Parameters
        ----------
        name : str
            Unique, non-empty name for this force contribution.
        force : VectorField
            The force field, attached to ``self.mesh``, with one component
            per spatial dimension of the mesh.
        units : str
            Human-readable units for ``force``. Defaults to ``"N/m^3"``.
        metadata : dict, optional
            Arbitrary additional bookkeeping for this contribution.

        Returns
        -------
        ForceContribution
            The newly registered contribution.

        Raises
        ------
        TypeError
            If ``force`` is not a VectorField, or ``name``/``units`` is not
            a string, or ``metadata`` is given and is not a dict.
        ValueError
            If ``name`` or ``units`` is empty/whitespace-only, if a
            contribution is already registered under ``name``, if ``force``
            is not attached to ``self.mesh``, or if ``force`` does not have
            one component per spatial dimension of ``self.mesh``.
        """
        contribution = ForceContribution(name, force, units, metadata)
        return self.register_contribution(contribution)

    def register_contribution(self, contribution: ForceContribution) -> ForceContribution:
        """Register an already-built :class:`ForceContribution`.

        Raises
        ------
        TypeError
            If ``contribution`` is not a ForceContribution.
        ValueError
            If a contribution is already registered under its name, if its
            force field is not attached to ``self.mesh``, or if it does not
            have one component per spatial dimension of ``self.mesh``.
        """
        if not isinstance(contribution, ForceContribution):
            raise TypeError(f"contribution must be a ForceContribution, got {type(contribution).__name__}.")
        if contribution.name in self._contributions:
            raise ValueError(f"a force contribution named '{contribution.name}' is already registered.")
        self._validate_mesh_compatibility(contribution.force)

        self._contributions[contribution.name] = contribution
        return contribution

    def _validate_mesh_compatibility(self, force: VectorField) -> None:
        if force.mesh is not self.mesh:
            raise ValueError("force field must be attached to the same mesh as this ForceAccounting instance.")
        expected_components = self.mesh.cell_centers.shape[1]
        actual_components = force.values.shape[1]
        if actual_components != expected_components:
            raise ValueError(
                f"force field must have {expected_components} component(s) to match the mesh "
                f"dimension, got {actual_components}."
            )

    # -- retrieval -----------------------------------------------------------

    @property
    def names(self) -> List[str]:
        """Names of every registered force contribution, in registration order."""
        return list(self._contributions.keys())

    def __len__(self) -> int:
        return len(self._contributions)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._contributions

    def get(self, name: str) -> ForceContribution:
        """Return the registered contribution with the given name.

        Raises
        ------
        TypeError
            If ``name`` is not a string.
        KeyError
            If no contribution with that name is registered.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}.")
        try:
            return self._contributions[name]
        except KeyError as exc:
            raise KeyError(f"no force contribution named '{name}' is registered.") from exc

    def get_force(self, name: str) -> VectorField:
        """Return the force field of the registered contribution with the given name."""
        return self.get(name).force

    # -- aggregation ---------------------------------------------------------

    def total_force(self) -> VectorField:
        """Return the per-cell sum of every registered force contribution.

        Returns
        -------
        VectorField
            The vector sum of all registered forces at every cell.

        Raises
        ------
        ValueError
            If no force contributions are registered.
        """
        if not self._contributions:
            raise ValueError("no force contributions are registered.")
        n_components = self.mesh.cell_centers.shape[1]
        total = np.zeros((self.mesh.n_cells, n_components))
        for contribution in self._contributions.values():
            total += contribution.force.values
        return VectorField(self.mesh, total)

    def magnitude(self, name: str) -> ScalarField:
        """Return the per-cell magnitude of the named force contribution."""
        return self.get(name).magnitude()

    def magnitudes(self) -> Dict[str, ScalarField]:
        """Return every registered contribution's per-cell magnitude, keyed by name."""
        return {name: contribution.magnitude() for name, contribution in self._contributions.items()}

    def total_magnitude(self) -> ScalarField:
        """Return the per-cell magnitude of :meth:`total_force`."""
        total = self.total_force()
        magnitude = np.sqrt(np.sum(total.values ** 2, axis=1))
        return ScalarField(self.mesh, magnitude)

    def normalized_contributions(self) -> Dict[str, ScalarField]:
        """Return each contribution's per-cell share of the summed force magnitudes.

        Each contribution's magnitude is divided by the *sum of every
        contribution's own magnitude* (not the magnitude of the vector
        total, which can be near zero even when individual forces are large
        and simply cancel). Away from cells where every force vanishes, the
        returned fractions sum to one at every cell.

        Returns
        -------
        dict of str to ScalarField
            Each contribution's per-cell normalized share, keyed by name.

        Raises
        ------
        ValueError
            If no force contributions are registered.
        """
        if not self._contributions:
            raise ValueError("no force contributions are registered.")
        magnitudes = self.magnitudes()
        stacked = np.stack([field.values for field in magnitudes.values()], axis=0)
        denominator = np.maximum(np.sum(stacked, axis=0), _MIN_MAGNITUDE)
        return {
            name: ScalarField(self.mesh, field.values / denominator)
            for name, field in magnitudes.items()
        }

    # -- reporting -------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, object]]:
        """Return per-contribution summary statistics, keyed by name.

        Each entry contains ``"units"``, ``"mean_magnitude"``,
        ``"max_magnitude"``, ``"min_magnitude"``,
        ``"mean_normalized_contribution"`` and a copy of ``"metadata"``.

        Raises
        ------
        ValueError
            If no force contributions are registered.
        """
        normalized = self.normalized_contributions()
        report = {}
        for name, contribution in self._contributions.items():
            magnitude = contribution.magnitude()
            report[name] = {
                "units": contribution.units,
                "mean_magnitude": magnitude.mean(),
                "max_magnitude": magnitude.max(),
                "min_magnitude": magnitude.min(),
                "mean_normalized_contribution": normalized[name].mean(),
                "metadata": dict(contribution.metadata),
            }
        return report

    def report(self) -> str:
        """Return a human-readable, multi-line summary of every contribution."""
        if not self._contributions:
            return f"ForceAccounting: no force contributions registered ({self.mesh.n_cells} cells)."

        summary = self.summary()
        header = f"{'name':<24}{'units':<12}{'mean |F|':>14}{'max |F|':>14}{'mean share':>14}"
        lines = [
            f"ForceAccounting: {len(self._contributions)} contribution(s), {self.mesh.n_cells} cells",
            header,
            "-" * len(header),
        ]
        for name, stats in summary.items():
            lines.append(
                f"{name:<24}{stats['units']:<12}{stats['mean_magnitude']:>14.4e}"
                f"{stats['max_magnitude']:>14.4e}{stats['mean_normalized_contribution']:>14.2%}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ForceAccounting(n_contributions={len(self._contributions)}, names={self.names!r})"
