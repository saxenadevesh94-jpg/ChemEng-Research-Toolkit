import numpy as np
import pytest

from src.cfd import FixedValueBC, ScalarField, VectorField
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.multiphase import EulerianMultiphaseSystem, Phase
from src.cfd.pulse_tracking import LiquidHoldupCalculator, PulseTracker
from src.cfd.transient_eulerian_solver import (
    TransientEulerianTwoFluidResult,
    solve_transient_eulerian_two_fluid,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# nx=5 gives x-positions 0.0, 0.25, 0.5, 0.75, 1.0; the middle row (j=1) cells
# at i=0, 2, 4 sit exactly at x=0.0 (inlet), x=0.5 (mid) and x=1.0 (outlet).
_NX, _NY = 5, 3
_INLET_CELL, _MID_CELL, _OUTLET_CELL = 5, 7, 9

# A synthetic pulse travelling from inlet -> mid -> outlet at a constant
# speed of 5.0 (distance 0.5 every 0.1 time units), attenuating as it goes:
# inlet amplitude 1.0, mid amplitude 0.8 (20% attenuation), outlet amplitude
# 0.5 (50% attenuation).
_TIME_POINTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
_INLET_SIGNAL = [0.0, 0.2, 1.0, 0.4, 0.1, 0.0]
_MID_SIGNAL = [0.0, 0.0, 0.3, 0.8, 0.3, 0.0]
_OUTLET_SIGNAL = [0.0, 0.0, 0.0, 0.2, 0.5, 0.2]


def _mesh():
    return build_structured_mesh(nx=_NX, ny=_NY)


def _pulse_result(mesh, liquid_fraction=0.6):
    n_cells = mesh.n_cells
    velocity_history = {"gas": [], "liquid": []}
    for step in range(len(_TIME_POINTS)):
        values = np.zeros((n_cells, 2))
        values[_INLET_CELL, 0] = _INLET_SIGNAL[step]
        values[_MID_CELL, 0] = _MID_SIGNAL[step]
        values[_OUTLET_CELL, 0] = _OUTLET_SIGNAL[step]
        velocity_history["liquid"].append(VectorField(mesh, values))
        velocity_history["gas"].append(VectorField(mesh, np.zeros((n_cells, 2))))

    pressure_history = [ScalarField(mesh, np.zeros(n_cells)) for _ in _TIME_POINTS]
    volume_fraction_history = {
        "liquid": [ScalarField(mesh, np.full(n_cells, liquid_fraction)) for _ in _TIME_POINTS],
        "gas": [ScalarField(mesh, np.full(n_cells, 1.0 - liquid_fraction)) for _ in _TIME_POINTS],
    }
    drag_coefficient_history = [ScalarField(mesh, np.zeros(n_cells)) for _ in _TIME_POINTS]

    return TransientEulerianTwoFluidResult(
        time_points=list(_TIME_POINTS),
        velocity_history=velocity_history,
        pressure_history=pressure_history,
        volume_fraction_history=volume_fraction_history,
        drag_coefficient_history=drag_coefficient_history,
        dt=0.1,
        total_time=0.5,
        n_steps=5,
        iterations_per_step=[0] * len(_TIME_POINTS),
        converged_per_step=[True] * len(_TIME_POINTS),
        residual_history_per_step=[[] for _ in _TIME_POINTS],
    )


def _locations():
    return {"inlet": (0.0, 0.5), "mid": (0.5, 0.5), "outlet": (1.0, 0.5)}


def _tracker(**kwargs):
    mesh = _mesh()
    result = _pulse_result(mesh)
    return PulseTracker(result, "liquid", _locations(), **kwargs)


# ---------------------------------------------------------------------------
# LiquidHoldupCalculator
# ---------------------------------------------------------------------------

def test_holdup_calculator_rejects_non_result():
    with pytest.raises(TypeError):
        LiquidHoldupCalculator("not a result", "liquid")


def test_holdup_calculator_rejects_non_string_phase():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(TypeError):
        LiquidHoldupCalculator(result, 123)


def test_holdup_calculator_rejects_unknown_phase():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(ValueError):
        LiquidHoldupCalculator(result, "solid")


def test_local_holdup_returns_stored_field():
    mesh = _mesh()
    result = _pulse_result(mesh, liquid_fraction=0.7)
    calculator = LiquidHoldupCalculator(result, "liquid")
    assert np.allclose(calculator.local_holdup(step=0).values, 0.7)
    assert np.allclose(calculator.local_holdup(step=-1).values, 0.7)


def test_local_holdup_step_out_of_range_raises_index_error():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    with pytest.raises(IndexError):
        calculator.local_holdup(step=100)


def test_local_holdup_history_length():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    assert len(calculator.local_holdup_history()) == len(_TIME_POINTS)


def test_local_holdup_at_rejects_bad_location_type():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    with pytest.raises(TypeError):
        calculator.local_holdup_at("not a location")


def test_local_holdup_at_rejects_wrong_length_location():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    with pytest.raises(ValueError):
        calculator.local_holdup_at((0.0, 0.0, 0.0))


def test_local_holdup_at_finds_nearest_cell():
    mesh = _mesh()
    result = _pulse_result(mesh, liquid_fraction=0.55)
    calculator = LiquidHoldupCalculator(result, "liquid")
    assert calculator.local_holdup_at((0.5, 0.5)) == pytest.approx(0.55)


def test_local_holdup_time_series_length_matches_history():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    series = calculator.local_holdup_time_series((0.5, 0.5))
    assert len(series) == len(_TIME_POINTS)


def test_global_holdup_matches_uniform_field_value():
    mesh = _mesh()
    result = _pulse_result(mesh, liquid_fraction=0.42)
    calculator = LiquidHoldupCalculator(result, "liquid")
    assert calculator.global_holdup(step=0) == pytest.approx(0.42)


def test_global_holdup_is_volume_weighted_average_for_nonuniform_field():
    mesh = _mesh()
    result = _pulse_result(mesh)
    non_uniform = np.linspace(0.1, 0.9, mesh.n_cells)
    result.volume_fraction_history["liquid"][0] = ScalarField(mesh, non_uniform)
    calculator = LiquidHoldupCalculator(result, "liquid")
    expected = float(np.sum(non_uniform * mesh.cell_volumes) / np.sum(mesh.cell_volumes))
    assert calculator.global_holdup(step=0) == pytest.approx(expected)


def test_global_holdup_history_length():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    assert len(calculator.global_holdup_history()) == len(_TIME_POINTS)


def test_holdup_calculator_repr_contains_phase_name():
    mesh = _mesh()
    result = _pulse_result(mesh)
    calculator = LiquidHoldupCalculator(result, "liquid")
    assert "liquid" in repr(calculator)


# ---------------------------------------------------------------------------
# PulseTracker: construction and validation
# ---------------------------------------------------------------------------

def test_pulse_tracker_rejects_non_result():
    with pytest.raises(TypeError):
        PulseTracker("not a result", "liquid", _locations())


def test_pulse_tracker_rejects_unknown_liquid_phase():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(ValueError):
        PulseTracker(result, "solid", _locations())


def test_pulse_tracker_rejects_invalid_signal():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(ValueError):
        PulseTracker(result, "liquid", _locations(), signal="not a signal")


@pytest.mark.parametrize("bad_fraction", [0.0, 1.5, -0.1, True])
def test_pulse_tracker_rejects_invalid_arrival_fraction(bad_fraction):
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises((TypeError, ValueError)):
        PulseTracker(result, "liquid", _locations(), arrival_fraction=bad_fraction)


def test_pulse_tracker_rejects_empty_monitoring_locations_dict():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(ValueError):
        PulseTracker(result, "liquid", {})


def test_pulse_tracker_rejects_empty_monitoring_locations_list():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(ValueError):
        PulseTracker(result, "liquid", [])


def test_pulse_tracker_rejects_non_mapping_non_sequence_locations():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(TypeError):
        PulseTracker(result, "liquid", 42)


def test_pulse_tracker_rejects_non_string_location_name():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(TypeError):
        PulseTracker(result, "liquid", {1: (0.0, 0.5)})


def test_pulse_tracker_rejects_wrong_length_coordinate():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(ValueError):
        PulseTracker(result, "liquid", {"inlet": (0.0, 0.5, 0.0)})


def test_pulse_tracker_rejects_non_sequence_coordinate():
    mesh = _mesh()
    result = _pulse_result(mesh)
    with pytest.raises(TypeError):
        PulseTracker(result, "liquid", {"inlet": "bad"})


def test_pulse_tracker_auto_names_list_of_coordinates():
    mesh = _mesh()
    result = _pulse_result(mesh)
    tracker = PulseTracker(result, "liquid", [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)])
    assert tracker.monitoring_names == ["point_0", "point_1", "point_2"]


def test_pulse_tracker_monitoring_names_preserve_insertion_order():
    tracker = _tracker()
    assert tracker.monitoring_names == ["inlet", "mid", "outlet"]


def test_pulse_tracker_resolved_location_matches_nearest_cell_center():
    tracker = _tracker()
    assert np.allclose(tracker.resolved_location("mid"), [0.5, 0.5])


def test_pulse_tracker_resolved_location_rejects_unknown_name():
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.resolved_location("unknown")


def test_pulse_tracker_signal_history_rejects_unknown_name():
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.signal_history("unknown")


def test_pulse_tracker_signal_history_matches_crafted_values():
    tracker = _tracker()
    assert np.allclose(tracker.signal_history("inlet"), _INLET_SIGNAL)
    assert np.allclose(tracker.signal_history("mid"), _MID_SIGNAL)
    assert np.allclose(tracker.signal_history("outlet"), _OUTLET_SIGNAL)


def test_pulse_tracker_time_points_match_result():
    tracker = _tracker()
    assert tracker.time_points == _TIME_POINTS


def test_pulse_tracker_global_holdup_history_delegates_to_calculator():
    mesh = _mesh()
    result = _pulse_result(mesh, liquid_fraction=0.6)
    tracker = PulseTracker(result, "liquid", _locations())
    assert np.allclose(tracker.global_holdup_history(), [0.6] * len(_TIME_POINTS))


# ---------------------------------------------------------------------------
# PulseTracker: baseline, peak
# ---------------------------------------------------------------------------

def test_baseline_value():
    tracker = _tracker()
    assert tracker.baseline_value("inlet") == pytest.approx(0.0)


def test_peak_value_and_peak_time():
    tracker = _tracker()
    assert tracker.peak_value("inlet") == pytest.approx(1.0)
    assert tracker.peak_time("inlet") == pytest.approx(0.2)
    assert tracker.peak_value("mid") == pytest.approx(0.8)
    assert tracker.peak_time("mid") == pytest.approx(0.3)
    assert tracker.peak_value("outlet") == pytest.approx(0.5)
    assert tracker.peak_time("outlet") == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# PulseTracker: arrival / departure / residence
# ---------------------------------------------------------------------------

def test_arrival_time_matches_expected_crossing():
    tracker = _tracker()
    assert tracker.arrival_time("inlet") == pytest.approx(0.1)
    assert tracker.arrival_time("mid") == pytest.approx(0.2)
    assert tracker.arrival_time("outlet") == pytest.approx(0.3)


def test_departure_time_matches_expected_crossing():
    tracker = _tracker()
    assert tracker.departure_time("inlet") == pytest.approx(0.5)
    assert tracker.departure_time("mid") == pytest.approx(0.5)
    # outlet's signal never drops back below its threshold within the
    # stored window, so departure falls back to the last stored time point.
    assert tracker.departure_time("outlet") == pytest.approx(0.5)


def test_residence_time_matches_departure_minus_arrival():
    tracker = _tracker()
    assert tracker.residence_time("inlet") == pytest.approx(0.4)
    assert tracker.residence_time("mid") == pytest.approx(0.3)
    assert tracker.residence_time("outlet") == pytest.approx(0.2)


def test_arrival_time_raises_when_no_pulse_detected():
    mesh = _mesh()
    result = _pulse_result(mesh)
    # Overwrite inlet's history with a flat signal (no pulse).
    for field in result.velocity_history["liquid"]:
        field.values[_INLET_CELL, 0] = 0.0
    tracker = PulseTracker(result, "liquid", _locations())
    with pytest.raises(ValueError):
        tracker.arrival_time("inlet")


def test_arrival_and_departure_accept_custom_fraction():
    tracker = _tracker()
    # A much larger fraction pushes the threshold closer to the peak, so
    # arrival should be later (or equal) and departure earlier (or equal)
    # than with the default fraction.
    default_arrival = tracker.arrival_time("inlet")
    strict_arrival = tracker.arrival_time("inlet", fraction=0.9)
    assert strict_arrival >= default_arrival


# ---------------------------------------------------------------------------
# PulseTracker: attenuation
# ---------------------------------------------------------------------------

def test_attenuation_matches_expected_amplitude_ratio():
    tracker = _tracker()
    assert tracker.attenuation("mid", "inlet") == pytest.approx(0.2)
    assert tracker.attenuation("outlet", "inlet") == pytest.approx(0.5)
    assert tracker.attenuation("inlet", "inlet") == pytest.approx(0.0)


def test_attenuation_rejects_reference_with_no_pulse():
    mesh = _mesh()
    result = _pulse_result(mesh)
    for field in result.velocity_history["liquid"]:
        field.values[_INLET_CELL, 0] = 0.0
    tracker = PulseTracker(result, "liquid", _locations())
    with pytest.raises(ValueError):
        tracker.attenuation("mid", "inlet")


# ---------------------------------------------------------------------------
# PulseTracker: pulse velocity
# ---------------------------------------------------------------------------

def test_pulse_velocity_using_arrival_matches_expected_speed():
    tracker = _tracker()
    assert tracker.pulse_velocity("inlet", "mid") == pytest.approx(5.0)
    assert tracker.pulse_velocity("inlet", "outlet") == pytest.approx(5.0)
    assert tracker.pulse_velocity("mid", "outlet") == pytest.approx(5.0)


def test_pulse_velocity_using_peak_matches_expected_speed():
    tracker = _tracker()
    assert tracker.pulse_velocity("mid", "outlet", using="peak") == pytest.approx(5.0)


def test_pulse_velocity_rejects_invalid_using():
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.pulse_velocity("inlet", "mid", using="not a mode")


def test_pulse_velocity_rejects_non_positive_elapsed_time():
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.pulse_velocity("mid", "inlet")


def test_pulse_velocity_rejects_same_location():
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.pulse_velocity("inlet", "inlet")


# ---------------------------------------------------------------------------
# PulseTracker: reporting
# ---------------------------------------------------------------------------

def test_summary_contains_all_locations_and_expected_keys():
    tracker = _tracker()
    summary = tracker.summary()
    assert set(summary.keys()) == {"inlet", "mid", "outlet"}
    for stats in summary.values():
        assert set(stats.keys()) == {
            "baseline_value", "peak_value", "peak_time", "arrival_time", "departure_time", "residence_time",
        }


def test_report_contains_every_location_name():
    tracker = _tracker()
    text = tracker.report()
    assert "inlet" in text
    assert "mid" in text
    assert "outlet" in text


def test_pulse_tracker_repr_contains_key_fields():
    tracker = _tracker()
    text = repr(tracker)
    assert "liquid" in text
    assert "velocity_magnitude" in text


# ---------------------------------------------------------------------------
# PulseTracker: holdup signal mode
# ---------------------------------------------------------------------------

def test_holdup_signal_tracks_volume_fraction_instead_of_velocity():
    mesh = _mesh()
    result = _pulse_result(mesh, liquid_fraction=0.5)
    n_cells = mesh.n_cells
    holdup_values = [0.5, 0.5, 0.6, 0.9, 0.6, 0.5]
    for step, value in enumerate(holdup_values):
        field = np.full(n_cells, 0.5)
        field[_INLET_CELL] = value
        result.volume_fraction_history["liquid"][step] = ScalarField(mesh, field)

    tracker = PulseTracker(result, "liquid", _locations(), signal="holdup")
    assert np.allclose(tracker.signal_history("inlet"), holdup_values)
    assert tracker.peak_value("inlet") == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Integration: real transient solve driving a pulsed inlet schedule
# ---------------------------------------------------------------------------

def _u_boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


def _v_boundaries():
    return [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
        FixedValueBC("bottom", 0.0),
    ]


def test_integration_with_real_pulsed_transient_solve():
    mesh = build_structured_mesh(nx=5, ny=5)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.3)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.7)
    system = EulerianMultiphaseSystem(mesh, [gas, liquid])

    def pulsed_top(time):
        # A short liquid pulse: velocity ramps up then back down.
        magnitude = 0.5 if time <= 0.02 else 0.0
        return _u_boundaries(top=magnitude)

    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.04,
        u_boundary_conditions=pulsed_top, v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5,
    )

    tracker = PulseTracker(result, "liquid", {"near_top": (0.5, 1.0), "near_bottom": (0.5, 0.0)})
    assert len(tracker.signal_history("near_top")) == len(result.time_points)
    assert np.all(np.isfinite(tracker.signal_history("near_top")))

    calculator = LiquidHoldupCalculator(result, "liquid")
    assert len(calculator.global_holdup_history()) == len(result.time_points)
    assert np.allclose(calculator.global_holdup_history(), 0.7)
