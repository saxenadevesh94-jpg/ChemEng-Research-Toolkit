"""Liquid holdup and pulse tracking for the transient Eulerian two-fluid solver.

Sprint 34 adds analysis tooling on top of the existing transient two-fluid
solver (:mod:`~src.cfd.transient_eulerian_solver`) rather than a new flow
solver. A liquid pulse is injected using infrastructure that already
exists in this package: :func:`~src.cfd.transient_eulerian_solver.solve_transient_eulerian_two_fluid`
already accepts a callable ``BoundaryConditionSchedule`` (``time -> boundary
conditions``), re-evaluated at every time step, precisely so that a
transient inlet condition - such as a short-duration liquid pulse riding on
a steady base flow - can be simulated. This module does not reimplement
that injection mechanism; it consumes the
:class:`~src.cfd.transient_eulerian_solver.TransientEulerianTwoFluidResult`
produced by a solve driven by such a schedule and extracts holdup and pulse
statistics from it.

Two classes are provided:

- :class:`LiquidHoldupCalculator` computes local (per-cell or per-point)
  and global (volume-averaged) liquid holdup from a stored result, reusing
  the existing :class:`~src.cfd.field.ScalarField` and
  :class:`~src.cfd.mesh.Mesh` infrastructure. Volume-fraction transport is
  not yet implemented upstream (see
  :mod:`~src.cfd.transient_eulerian_solver`), so holdup is currently
  constant over time; this calculator is still wired up against the full
  time history so it keeps working, unchanged, once transport lands in a
  later sprint.
- :class:`PulseTracker` follows a scalar tracer signal at a set of
  arbitrary monitoring locations and derives classic pulse-flow
  diagnostics from it: arrival time, peak time, residence time,
  attenuation and propagation velocity between monitoring points. The
  tracer signal defaults to the liquid phase's velocity magnitude - the
  field that actually responds to a pulsed inlet schedule today - but can
  be switched to liquid holdup once volume-fraction transport makes it a
  useful pulse signal too.

Keeping tracking separate from holdup computation, and both separate from
solving, keeps this module usable for later ERT (electrical resistance
tomography) comparison and validation without depending on, or
constraining, how the underlying flow was solved.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

from .field import ScalarField
from .mesh import Mesh
from .transient_eulerian_solver import TransientEulerianTwoFluidResult

_SIGNAL_VELOCITY_MAGNITUDE = "velocity_magnitude"
_SIGNAL_HOLDUP = "holdup"
_VALID_SIGNALS = (_SIGNAL_VELOCITY_MAGNITUDE, _SIGNAL_HOLDUP)

MonitoringLocations = Union[Dict[str, Sequence[float]], Sequence[Sequence[float]]]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_result(result: object) -> TransientEulerianTwoFluidResult:
    if not isinstance(result, TransientEulerianTwoFluidResult):
        raise TypeError(
            f"result must be a TransientEulerianTwoFluidResult, got {type(result).__name__}."
        )
    return result


def _validate_liquid_phase(result: TransientEulerianTwoFluidResult, liquid_phase: object) -> str:
    if not isinstance(liquid_phase, str):
        raise TypeError(f"liquid_phase must be a string, got {type(liquid_phase).__name__}.")
    if liquid_phase not in result.phase_names:
        raise ValueError(f"liquid_phase '{liquid_phase}' is not a phase in this result.")
    return liquid_phase


def _validate_location(location: object, mesh: Mesh) -> np.ndarray:
    if isinstance(location, str) or not isinstance(location, (list, tuple, np.ndarray)):
        raise TypeError(f"a monitoring location must be a sequence of numbers, got {type(location).__name__}.")
    array = np.asarray(location, dtype=float)
    n_dims = mesh.cell_centers.shape[1]
    if array.shape != (n_dims,):
        raise ValueError(f"a monitoring location must have exactly {n_dims} coordinate(s), got shape {array.shape}.")
    return array


def _validate_monitoring_locations(locations: object, mesh: Mesh) -> Dict[str, np.ndarray]:
    if isinstance(locations, dict):
        if not locations:
            raise ValueError("monitoring_locations must contain at least one location.")
        resolved: Dict[str, np.ndarray] = {}
        for name, location in locations.items():
            if not isinstance(name, str) or not name.strip():
                raise TypeError("every monitoring location name must be a non-empty string.")
            resolved[name] = _validate_location(location, mesh)
        return resolved

    if isinstance(locations, (list, tuple)):
        if not locations:
            raise ValueError("monitoring_locations must contain at least one location.")
        return {f"point_{i}": _validate_location(location, mesh) for i, location in enumerate(locations)}

    raise TypeError(
        "monitoring_locations must be a mapping of name to coordinates, or a sequence of coordinates, "
        f"got {type(locations).__name__}."
    )


def _nearest_cell_index(mesh: Mesh, location: np.ndarray) -> int:
    distances = np.sum((mesh.cell_centers - location) ** 2, axis=1)
    return int(np.argmin(distances))


def _validate_fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not (0.0 < number <= 1.0):
        raise ValueError(f"{name} must be within the range (0, 1].")
    return number


def _validate_signal(signal: object) -> str:
    if signal not in _VALID_SIGNALS:
        raise ValueError(f"signal must be one of {_VALID_SIGNALS!r}, got {signal!r}.")
    return signal


def _validate_name_present(name: object, known_names: Sequence[str], field_name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string, got {type(name).__name__}.")
    if name not in known_names:
        raise ValueError(f"no monitoring location named '{name}' ({field_name}).")
    return name


# ---------------------------------------------------------------------------
# LiquidHoldupCalculator
# ---------------------------------------------------------------------------

class LiquidHoldupCalculator:
    """Compute local and global liquid holdup from a transient two-fluid result.

    Parameters
    ----------
    result : TransientEulerianTwoFluidResult
        A stored transient two-fluid solve, as returned by
        :func:`~src.cfd.transient_eulerian_solver.solve_transient_eulerian_two_fluid`.
    liquid_phase : str
        Name of the liquid phase within ``result`` to report holdup for.

    Raises
    ------
    TypeError
        If ``result`` is not a TransientEulerianTwoFluidResult, or
        ``liquid_phase`` is not a string.
    ValueError
        If ``liquid_phase`` is not a phase in ``result``.
    """

    def __init__(self, result: TransientEulerianTwoFluidResult, liquid_phase: str) -> None:
        self.result = _validate_result(result)
        self.liquid_phase = _validate_liquid_phase(result, liquid_phase)
        self.mesh: Mesh = result.pressure(0).mesh

    @property
    def time_points(self) -> List[float]:
        """Simulation time at every stored point, including ``t=0``."""
        return list(self.result.time_points)

    def local_holdup(self, step: int = -1) -> ScalarField:
        """Return the liquid volume-fraction field at a stored time step.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        """
        return self.result.volume_fraction(self.liquid_phase, step)

    def local_holdup_history(self) -> List[ScalarField]:
        """Return the liquid volume-fraction field at every stored time step."""
        return list(self.result.volume_fraction_history[self.liquid_phase])

    def local_holdup_at(self, location: Sequence[float], step: int = -1) -> float:
        """Return local liquid holdup at the mesh cell nearest ``location``.

        Raises
        ------
        TypeError
            If ``location`` is not a sequence of numbers.
        ValueError
            If ``location`` does not have one coordinate per mesh dimension.
        IndexError
            If ``step`` is out of range.
        """
        array = _validate_location(location, self.mesh)
        cell_index = _nearest_cell_index(self.mesh, array)
        return float(self.local_holdup(step).values[cell_index])

    def local_holdup_time_series(self, location: Sequence[float]) -> List[float]:
        """Return local liquid holdup at the nearest cell to ``location``, at every stored time step."""
        array = _validate_location(location, self.mesh)
        cell_index = _nearest_cell_index(self.mesh, array)
        return [float(field.values[cell_index]) for field in self.local_holdup_history()]

    def global_holdup(self, step: int = -1) -> float:
        """Return the volume-averaged (domain-wide) liquid holdup at a stored time step.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        """
        field = self.local_holdup(step)
        volumes = self.mesh.cell_volumes
        return float(np.sum(field.values * volumes) / np.sum(volumes))

    def global_holdup_history(self) -> List[float]:
        """Return the volume-averaged liquid holdup at every stored time step."""
        return [self.global_holdup(step) for step in range(len(self.result.time_points))]

    def __repr__(self) -> str:
        return (
            f"LiquidHoldupCalculator(liquid_phase={self.liquid_phase!r}, "
            f"n_time_points={len(self.result.time_points)})"
        )


# ---------------------------------------------------------------------------
# PulseTracker
# ---------------------------------------------------------------------------

class PulseTracker:
    """Track a liquid pulse through a transient two-fluid result at monitoring locations.

    Parameters
    ----------
    result : TransientEulerianTwoFluidResult
        A stored transient two-fluid solve, typically produced with a
        pulsed/ramped inlet boundary-condition schedule (see the module
        docstring).
    liquid_phase : str
        Name of the liquid phase within ``result`` to track.
    monitoring_locations : mapping of str to coordinates, or sequence of coordinates
        Arbitrary points at which to track the pulse, given either as a
        mapping of name to coordinates (e.g.
        ``{"inlet": (0.05, 0.5), "outlet": (0.95, 0.5)}``) or as a plain
        sequence of coordinates, auto-named ``"point_0"``, ``"point_1"``,
        etc. Each location is resolved to its nearest mesh cell.
    signal : str
        Which per-cell field to use as the pulse tracer signal: either
        ``"velocity_magnitude"`` (the liquid phase's velocity magnitude -
        the default, since it responds to a pulsed inlet schedule today)
        or ``"holdup"`` (the liquid volume fraction, useful once
        volume-fraction transport is implemented upstream). Defaults to
        ``"velocity_magnitude"``.
    arrival_fraction : float
        Fraction (in ``(0, 1]``) of each location's own pulse amplitude
        (peak minus baseline) used as the crossing threshold for arrival
        and departure times. Defaults to ``0.1`` (10%), a common tracer
        threshold. Each location uses its own baseline and amplitude, so
        this fraction is comparable across locations even as the pulse
        attenuates.

    Raises
    ------
    TypeError
        If any input has the wrong type.
    ValueError
        If ``liquid_phase`` is unknown, ``signal`` is not a supported
        value, ``arrival_fraction`` is out of range, or
        ``monitoring_locations`` is empty or malformed.
    """

    def __init__(
        self,
        result: TransientEulerianTwoFluidResult,
        liquid_phase: str,
        monitoring_locations: MonitoringLocations,
        signal: str = _SIGNAL_VELOCITY_MAGNITUDE,
        arrival_fraction: float = 0.1,
    ) -> None:
        self.result = _validate_result(result)
        self.liquid_phase = _validate_liquid_phase(result, liquid_phase)
        self.signal = _validate_signal(signal)
        self.arrival_fraction = _validate_fraction(arrival_fraction, "arrival_fraction")

        self.holdup_calculator = LiquidHoldupCalculator(result, liquid_phase)
        self.mesh = self.holdup_calculator.mesh

        self._locations = _validate_monitoring_locations(monitoring_locations, self.mesh)
        self._cell_indices: Dict[str, int] = {
            name: _nearest_cell_index(self.mesh, location) for name, location in self._locations.items()
        }
        self._signal_history: Dict[str, List[float]] = self._build_signal_history()

    def _signal_field_values(self, step: int) -> np.ndarray:
        if self.signal == _SIGNAL_HOLDUP:
            return self.holdup_calculator.local_holdup(step).values
        velocity = self.result.velocity(self.liquid_phase, step)
        return np.sqrt(np.sum(velocity.values ** 2, axis=1))

    def _build_signal_history(self) -> Dict[str, List[float]]:
        n_points = len(self.result.time_points)
        matrix = np.stack([self._signal_field_values(step) for step in range(n_points)], axis=0)
        return {name: matrix[:, cell_index].tolist() for name, cell_index in self._cell_indices.items()}

    # -- lookups -------------------------------------------------------------

    @property
    def monitoring_names(self) -> List[str]:
        """Names of every monitoring location, in the order they were given."""
        return list(self._locations.keys())

    @property
    def time_points(self) -> List[float]:
        """Simulation time at every stored point, including ``t=0``."""
        return list(self.result.time_points)

    def resolved_location(self, name: str) -> np.ndarray:
        """Return the actual mesh cell-center coordinates used for a monitoring location."""
        _validate_name_present(name, self.monitoring_names, "name")
        return self.mesh.cell_centers[self._cell_indices[name]]

    def signal_history(self, name: str) -> List[float]:
        """Return the tracer signal's full time history at a monitoring location."""
        _validate_name_present(name, self.monitoring_names, "name")
        return list(self._signal_history[name])

    def global_holdup_history(self) -> List[float]:
        """Return the volume-averaged liquid holdup at every stored time step."""
        return self.holdup_calculator.global_holdup_history()

    # -- pulse metrics ---------------------------------------------------------

    def baseline_value(self, name: str) -> float:
        """Return the tracer signal's value at ``t=0`` at a monitoring location."""
        return self.signal_history(name)[0]

    def peak_value(self, name: str) -> float:
        """Return the tracer signal's maximum value at a monitoring location."""
        return float(np.max(self.signal_history(name)))

    def peak_time(self, name: str) -> float:
        """Return the simulation time at which the tracer signal peaks at a monitoring location."""
        history = self.signal_history(name)
        return self.time_points[int(np.argmax(history))]

    def _amplitude(self, name: str) -> float:
        return self.peak_value(name) - self.baseline_value(name)

    def _threshold(self, name: str, fraction: float) -> float:
        return self.baseline_value(name) + fraction * self._amplitude(name)

    def arrival_time(self, name: str, fraction: float = None) -> float:
        """Return the first time the pulse crosses its arrival threshold at a monitoring location.

        The threshold is ``baseline + fraction * (peak - baseline)``, using
        that location's own baseline and peak.

        Raises
        ------
        ValueError
            If the tracer signal never rises above its baseline at this
            location (no pulse detected), or ``fraction`` is out of range.
        """
        fraction = self.arrival_fraction if fraction is None else _validate_fraction(fraction, "fraction")
        amplitude = self._amplitude(name)
        if amplitude <= 0.0:
            raise ValueError(f"no pulse detected at monitoring location '{name}' (signal never rises above baseline).")

        threshold = self._threshold(name, fraction)
        history = self.signal_history(name)
        peak_index = int(np.argmax(history))
        for index in range(peak_index + 1):
            if history[index] >= threshold:
                return self.time_points[index]
        return self.time_points[peak_index]

    def departure_time(self, name: str, fraction: float = None) -> float:
        """Return the first time (at or after the peak) the pulse drops back below its threshold.

        If the tracer signal never drops back below the threshold within
        the stored history, the last stored time point is returned.

        Raises
        ------
        ValueError
            If the tracer signal never rises above its baseline at this
            location (no pulse detected), or ``fraction`` is out of range.
        """
        fraction = self.arrival_fraction if fraction is None else _validate_fraction(fraction, "fraction")
        amplitude = self._amplitude(name)
        if amplitude <= 0.0:
            raise ValueError(f"no pulse detected at monitoring location '{name}' (signal never rises above baseline).")

        threshold = self._threshold(name, fraction)
        history = self.signal_history(name)
        peak_index = int(np.argmax(history))
        for index in range(peak_index, len(history)):
            if history[index] < threshold:
                return self.time_points[index]
        return self.time_points[-1]

    def residence_time(self, name: str, fraction: float = None) -> float:
        """Return how long the pulse stays above its threshold at a monitoring location.

        Equal to :meth:`departure_time` minus :meth:`arrival_time`.
        """
        return self.departure_time(name, fraction) - self.arrival_time(name, fraction)

    def attenuation(self, name: str, reference_name: str) -> float:
        """Return the fractional loss of pulse amplitude at ``name`` relative to ``reference_name``.

        ``0`` means no attenuation (equal amplitude), ``1`` means the pulse
        has fully died out by ``name``, and a negative value means the
        pulse amplified rather than attenuated.

        Raises
        ------
        ValueError
            If ``reference_name`` has no detectable pulse (non-positive
            amplitude).
        """
        reference_amplitude = self._amplitude(reference_name)
        if reference_amplitude <= 0.0:
            raise ValueError(f"no pulse detected at reference location '{reference_name}' (cannot normalise).")
        return 1.0 - (self._amplitude(name) / reference_amplitude)

    def pulse_velocity(self, from_name: str, to_name: str, using: str = "arrival") -> float:
        """Return the pulse's average propagation speed between two monitoring locations.

        Computed as the Euclidean distance between the two locations'
        resolved cell centers, divided by the difference in their arrival
        (or peak) times.

        Parameters
        ----------
        from_name, to_name : str
            Names of the upstream and downstream monitoring locations.
        using : str
            Either ``"arrival"`` (default) or ``"peak"``, selecting which
            timing metric to use.

        Raises
        ------
        ValueError
            If ``using`` is not ``"arrival"`` or ``"peak"``, or the pulse
            does not arrive (or peak) strictly later at ``to_name`` than at
            ``from_name``.
        """
        if using == "arrival":
            t_from, t_to = self.arrival_time(from_name), self.arrival_time(to_name)
        elif using == "peak":
            t_from, t_to = self.peak_time(from_name), self.peak_time(to_name)
        else:
            raise ValueError("using must be either 'arrival' or 'peak'.")

        elapsed = t_to - t_from
        if elapsed <= 0.0:
            raise ValueError(
                f"pulse must reach '{to_name}' strictly later than '{from_name}' to compute a velocity "
                f"(got elapsed time {elapsed})."
            )
        distance = float(np.linalg.norm(self.resolved_location(to_name) - self.resolved_location(from_name)))
        return distance / elapsed

    # -- reporting -------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return every pulse metric at every monitoring location, keyed by location name.

        Each entry contains ``"baseline_value"``, ``"peak_value"``,
        ``"peak_time"``, ``"arrival_time"``, ``"departure_time"`` and
        ``"residence_time"``.

        Raises
        ------
        ValueError
            If any monitoring location has no detectable pulse.
        """
        report: Dict[str, Dict[str, float]] = {}
        for name in self.monitoring_names:
            report[name] = {
                "baseline_value": self.baseline_value(name),
                "peak_value": self.peak_value(name),
                "peak_time": self.peak_time(name),
                "arrival_time": self.arrival_time(name),
                "departure_time": self.departure_time(name),
                "residence_time": self.residence_time(name),
            }
        return report

    def report(self) -> str:
        """Return a human-readable, multi-line summary of every monitoring location's pulse metrics."""
        header = f"{'location':<16}{'arrival':>12}{'peak time':>12}{'peak value':>14}{'residence':>12}"
        lines = [
            f"PulseTracker: {len(self.monitoring_names)} location(s), signal={self.signal!r}",
            header,
            "-" * len(header),
        ]
        for name, stats in self.summary().items():
            lines.append(
                f"{name:<16}{stats['arrival_time']:>12.4g}{stats['peak_time']:>12.4g}"
                f"{stats['peak_value']:>14.4g}{stats['residence_time']:>12.4g}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PulseTracker(liquid_phase={self.liquid_phase!r}, signal={self.signal!r}, "
            f"monitoring_names={self.monitoring_names!r})"
        )
