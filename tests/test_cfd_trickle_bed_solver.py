import numpy as np
import pytest

from src.cfd import FixedValueBC, ScalarField, VectorField, ZeroGradientBC
from src.cfd.capillary_pressure import ConstantCapillaryPressureModel, LeverettJFunctionModel
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.force_accounting import ForceAccounting
from src.cfd.multiphase import EulerianMultiphaseSystem, Phase
from src.cfd.packed_bed import PackedBedProperties
from src.cfd.pulse_tracking import LiquidHoldupCalculator, PulseTracker
from src.cfd.transient_eulerian_solver import TransientEulerianTwoFluidResult
from src.cfd.trickle_bed_solver import (
    ConstantRelativePermeabilityModel,
    CoreyRelativePermeabilityModel,
    RelativePermeabilityModel,
    TrickleBedResult,
    assemble_trickle_bed_phase_momentum_system,
    assemble_transient_trickle_bed_phase_momentum_system,
    relative_permeability_closure_equation,
    solve_trickle_bed,
    trickle_bed_momentum_equation,
)


def _u_boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


def _v_boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


def _gas_liquid_system(mesh, liquid_fraction=0.3):
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=1.0 - liquid_fraction)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=liquid_fraction)
    return EulerianMultiphaseSystem(mesh, [gas, liquid])


def _bed_properties():
    return PackedBedProperties(particle_diameter=5e-3, porosity=0.4, fluid_viscosity=1e-3, fluid_density=1000.0)


def _capillary_model(mesh):
    return ConstantCapillaryPressureModel(mesh, capillary_pressure_value=10.0)


def _kr_model(mesh, liquid_exponent=2.0, gas_exponent=2.0):
    return CoreyRelativePermeabilityModel(mesh, liquid_exponent=liquid_exponent, gas_exponent=gas_exponent)


def _solve(
    mesh=None,
    liquid_fraction=0.3,
    capillary_model=None,
    kr_model=None,
    dt=0.01,
    total_time=0.02,
    u_boundary_conditions=None,
    v_boundary_conditions=None,
    max_outer_iterations=5,
    mixture_max_outer_iterations=5,
    **kwargs,
):
    mesh = mesh if mesh is not None else build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh, liquid_fraction=liquid_fraction)
    capillary_model = capillary_model if capillary_model is not None else _capillary_model(mesh)
    kr_model = kr_model if kr_model is not None else _kr_model(mesh)
    u_boundary_conditions = u_boundary_conditions if u_boundary_conditions is not None else _u_boundaries()
    v_boundary_conditions = v_boundary_conditions if v_boundary_conditions is not None else _v_boundaries()
    return solve_trickle_bed(
        system,
        "liquid",
        particle_diameter=1e-3,
        bed_properties=_bed_properties(),
        capillary_pressure_model=capillary_model,
        relative_permeability_model=kr_model,
        surface_tension=0.07,
        contact_angle=0.5,
        dt=dt,
        total_time=total_time,
        u_boundary_conditions=u_boundary_conditions,
        v_boundary_conditions=v_boundary_conditions,
        max_outer_iterations=max_outer_iterations,
        mixture_max_outer_iterations=mixture_max_outer_iterations,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Descriptive bookkeeping
# ---------------------------------------------------------------------------

def test_trickle_bed_momentum_equation_contains_expected_terms():
    equation = trickle_bed_momentum_equation("liquid", "gas")
    text = str(equation)
    assert "ddt(U_liquid)" in text
    assert "U_gas" in text
    assert "drag" in text
    assert "ergun_resistance" in text
    assert "kr_liquid" in text


def test_trickle_bed_momentum_equation_rejects_empty_phase_name():
    with pytest.raises(ValueError):
        trickle_bed_momentum_equation("", "gas")


def test_trickle_bed_momentum_equation_rejects_empty_other_phase_name():
    with pytest.raises(ValueError):
        trickle_bed_momentum_equation("liquid", "")


def test_relative_permeability_closure_equation_contains_phase_name():
    equation = relative_permeability_closure_equation("liquid")
    text = str(equation)
    assert "kr_liquid" in text
    assert "S_liquid" in text


def test_relative_permeability_closure_equation_rejects_empty_name():
    with pytest.raises(ValueError):
        relative_permeability_closure_equation("")


# ---------------------------------------------------------------------------
# RelativePermeabilityModel
# ---------------------------------------------------------------------------

def test_relative_permeability_model_rejects_non_mesh():
    with pytest.raises(TypeError):
        CoreyRelativePermeabilityModel("not a mesh")


def test_relative_permeabilities_rejects_non_scalar_field():
    mesh = build_structured_mesh(nx=5, ny=5)
    model = _kr_model(mesh)
    with pytest.raises(TypeError):
        model.relative_permeabilities("not a field")


def test_relative_permeabilities_rejects_field_on_different_mesh():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    model = _kr_model(mesh)
    saturation = ScalarField(other_mesh, np.full(other_mesh.n_cells, 0.5))
    with pytest.raises(ValueError):
        model.relative_permeabilities(saturation)


def test_relative_permeabilities_rejects_out_of_range_saturation():
    mesh = build_structured_mesh(nx=5, ny=5)
    model = _kr_model(mesh)
    saturation = ScalarField(mesh, np.full(mesh.n_cells, 1.5))
    with pytest.raises(ValueError):
        model.relative_permeabilities(saturation)


def test_corey_relative_permeability_endpoints():
    mesh = build_structured_mesh(nx=5, ny=5)
    model = _kr_model(mesh)
    saturated = ScalarField(mesh, np.full(mesh.n_cells, 1.0))
    dry = ScalarField(mesh, np.full(mesh.n_cells, 0.0))

    kr_liquid, kr_gas = model.relative_permeabilities(saturated)
    assert np.allclose(kr_liquid.values, 1.0)
    assert np.allclose(kr_gas.values, 0.0)

    kr_liquid, kr_gas = model.relative_permeabilities(dry)
    assert np.allclose(kr_liquid.values, 0.0)
    assert np.allclose(kr_gas.values, 1.0)


def test_corey_relative_permeability_midpoint_matches_power_law():
    mesh = build_structured_mesh(nx=5, ny=5)
    model = CoreyRelativePermeabilityModel(mesh, liquid_exponent=2.0, gas_exponent=3.0)
    saturation = ScalarField(mesh, np.full(mesh.n_cells, 0.4))
    kr_liquid, kr_gas = model.relative_permeabilities(saturation)
    assert np.allclose(kr_liquid.values, 0.4 ** 2.0)
    assert np.allclose(kr_gas.values, 0.6 ** 3.0)


@pytest.mark.parametrize("bad_value", [True, "2.0", None, 0.0, -1.0])
def test_corey_relative_permeability_rejects_invalid_exponent(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        CoreyRelativePermeabilityModel(mesh, liquid_exponent=bad_value)


def test_constant_relative_permeability_returns_given_values():
    mesh = build_structured_mesh(nx=5, ny=5)
    model = ConstantRelativePermeabilityModel(mesh, liquid_relative_permeability=0.2, gas_relative_permeability=0.9)
    saturation = ScalarField(mesh, np.full(mesh.n_cells, 0.5))
    kr_liquid, kr_gas = model.relative_permeabilities(saturation)
    assert np.allclose(kr_liquid.values, 0.2)
    assert np.allclose(kr_gas.values, 0.9)


@pytest.mark.parametrize("bad_value", [True, "1.0", None, 1.5, -0.1])
def test_constant_relative_permeability_rejects_out_of_range_value(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        ConstantRelativePermeabilityModel(mesh, liquid_relative_permeability=bad_value)


def test_relative_permeability_model_is_abstract():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        RelativePermeabilityModel(mesh)


# ---------------------------------------------------------------------------
# assemble_trickle_bed_phase_momentum_system
# ---------------------------------------------------------------------------

def _phase_momentum_inputs(mesh):
    phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.3)
    velocity = VectorField(mesh, np.full((mesh.n_cells, 2), [0.1, 0.0]))
    other_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    drag = ScalarField(mesh, np.zeros(mesh.n_cells))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    kr = ScalarField(mesh, np.full(mesh.n_cells, 0.5))
    return phase, velocity, other_velocity, drag, pressure, kr


def test_assemble_trickle_bed_phase_momentum_system_returns_correctly_sized_system():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, kr = _phase_momentum_inputs(mesh)
    system = assemble_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, _bed_properties(), kr, _u_boundaries(), "u"
    )
    assert system.size == mesh.n_cells


def test_assemble_trickle_bed_phase_momentum_system_rejects_bad_bed_properties():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, kr = _phase_momentum_inputs(mesh)
    with pytest.raises(TypeError):
        assemble_trickle_bed_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, "not bed properties", kr, _u_boundaries(), "u"
        )


def test_assemble_trickle_bed_phase_momentum_system_rejects_bad_component():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, kr = _phase_momentum_inputs(mesh)
    with pytest.raises(ValueError):
        assemble_trickle_bed_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, _bed_properties(), kr, _u_boundaries(), "w"
        )


def test_assemble_trickle_bed_phase_momentum_system_rejects_bad_relative_permeability_type():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, _ = _phase_momentum_inputs(mesh)
    with pytest.raises(TypeError):
        assemble_trickle_bed_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, _bed_properties(), "not a field",
            _u_boundaries(), "u",
        )


def test_lower_relative_permeability_increases_diagonal_resistance():
    # A smaller relative permeability divides the Ergun term by a smaller
    # number, so it should add a strictly larger term to the diagonal.
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, _ = _phase_momentum_inputs(mesh)
    high_kr = ScalarField(mesh, np.full(mesh.n_cells, 0.9))
    low_kr = ScalarField(mesh, np.full(mesh.n_cells, 0.1))

    system_high_kr = assemble_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, _bed_properties(), high_kr, _u_boundaries(), "u"
    )
    system_low_kr = assemble_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, _bed_properties(), low_kr, _u_boundaries(), "u"
    )

    interior_cell = mesh.n_cells // 2
    assert system_low_kr.matrix.get(interior_cell, interior_cell) > system_high_kr.matrix.get(
        interior_cell, interior_cell
    )


# ---------------------------------------------------------------------------
# assemble_transient_trickle_bed_phase_momentum_system
# ---------------------------------------------------------------------------

def test_assemble_transient_trickle_bed_phase_momentum_system_rejects_non_positive_dt():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, kr = _phase_momentum_inputs(mesh)
    previous_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        assemble_transient_trickle_bed_phase_momentum_system(
            mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _bed_properties(), kr,
            _u_boundaries(), "u", dt=0.0,
        )


def test_assemble_transient_trickle_bed_phase_momentum_system_diagonal_grows_as_dt_shrinks():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure, kr = _phase_momentum_inputs(mesh)
    previous_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))

    system_large_dt = assemble_transient_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _bed_properties(), kr,
        _u_boundaries(), "u", dt=1.0,
    )
    system_small_dt = assemble_transient_trickle_bed_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _bed_properties(), kr,
        _u_boundaries(), "u", dt=0.01,
    )

    interior_cell = mesh.n_cells // 2
    assert system_small_dt.matrix.get(interior_cell, interior_cell) > system_large_dt.matrix.get(
        interior_cell, interior_cell
    )


# ---------------------------------------------------------------------------
# solve_trickle_bed: basic solving
# ---------------------------------------------------------------------------

def test_zero_boundary_conditions_give_near_zero_velocity_for_both_phases():
    result = _solve()
    assert result.converged is True
    assert np.allclose(result.velocity("liquid").values, 0.0, atol=1e-6)
    assert np.allclose(result.velocity("gas").values, 0.0, atol=1e-6)


def test_result_is_trickle_bed_result_and_transient_result():
    result = _solve()
    assert isinstance(result, TrickleBedResult)
    assert isinstance(result, TransientEulerianTwoFluidResult)


def test_history_length_matches_number_of_time_steps():
    result = _solve(dt=0.01, total_time=0.03)
    assert result.n_steps == 3
    assert len(result.time_points) == 4
    for name in ("liquid", "gas"):
        assert len(result.velocity_history[name]) == 4
        assert len(result.volume_fraction_history[name]) == 4
    assert len(result.pressure_history) == 4


def test_initial_history_entry_matches_system_initial_condition():
    result = _solve(liquid_fraction=0.25)
    assert np.allclose(result.volume_fraction("liquid", step=0).values, 0.25)
    assert np.allclose(result.volume_fraction("gas", step=0).values, 0.75)


def test_driven_case_runs_without_error_and_stays_finite():
    result = _solve(
        mesh=build_structured_mesh(nx=6, ny=6),
        u_boundary_conditions=_u_boundaries(top=0.05),
    )
    for name in ("liquid", "gas"):
        velocity = result.velocity(name)
        assert velocity.values.shape[1] == 2
        assert np.all(np.isfinite(velocity.values))
    assert np.all(np.isfinite(result.pressure().values))


def test_callable_boundary_condition_schedule_varies_with_time():
    seen_times = []

    def ramped_top(time):
        seen_times.append(time)
        return _u_boundaries(top=time)

    result = _solve(u_boundary_conditions=ramped_top)
    assert result.n_steps == 2
    assert np.allclose(seen_times, [0.01, 0.02])


def test_low_relative_permeability_reduces_liquid_velocity_versus_full_permeability():
    mesh = build_structured_mesh(nx=6, ny=6)
    full_kr = ConstantRelativePermeabilityModel(mesh, 1.0, 1.0)
    low_kr = ConstantRelativePermeabilityModel(mesh, 0.05, 1.0)

    result_full = _solve(
        mesh=mesh, kr_model=full_kr, u_boundary_conditions=_u_boundaries(top=0.05),
    )
    result_low = _solve(
        mesh=mesh, kr_model=low_kr, u_boundary_conditions=_u_boundaries(top=0.05),
    )

    liquid_speed_full = np.max(np.abs(result_full.velocity("liquid").values))
    liquid_speed_low = np.max(np.abs(result_low.velocity("liquid").values))
    assert liquid_speed_low <= liquid_speed_full


# ---------------------------------------------------------------------------
# solve_trickle_bed: holdup and pulse tracking
# ---------------------------------------------------------------------------

def test_holdup_calculator_is_attached_automatically():
    result = _solve(liquid_fraction=0.4)
    assert isinstance(result.holdup_calculator, LiquidHoldupCalculator)
    assert result.holdup_calculator.global_holdup() == pytest.approx(0.4)


def test_pulse_tracker_is_none_without_monitoring_locations():
    result = _solve()
    assert result.pulse_tracker is None


def test_pulse_tracker_is_attached_with_monitoring_locations():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = _solve(mesh=mesh, monitoring_locations={"inlet": (0.0, 0.5), "outlet": (1.0, 0.5)})
    assert isinstance(result.pulse_tracker, PulseTracker)
    assert set(result.pulse_tracker.monitoring_names) == {"inlet", "outlet"}


def test_attach_pulse_tracker_can_be_called_manually():
    result = _solve()
    tracker = result.attach_pulse_tracker({"point": (0.5, 0.5)})
    assert result.pulse_tracker is tracker
    assert isinstance(tracker, PulseTracker)


# ---------------------------------------------------------------------------
# solve_trickle_bed: force accounting
# ---------------------------------------------------------------------------

def test_build_force_accounting_returns_expected_contributions_for_liquid():
    result = _solve()
    accounting = result.build_force_accounting("liquid")
    assert isinstance(accounting, ForceAccounting)
    assert "pressure_gradient" in accounting
    assert "interphase_drag" in accounting
    assert "packed_bed_resistance" in accounting
    assert "capillary_pressure" in accounting


def test_build_force_accounting_excludes_capillary_pressure_for_gas():
    result = _solve()
    accounting = result.build_force_accounting("gas")
    assert "packed_bed_resistance" in accounting
    assert "capillary_pressure" not in accounting


def test_build_force_accounting_rejects_unknown_phase():
    result = _solve()
    with pytest.raises(KeyError):
        result.build_force_accounting("solid")


def test_relative_permeability_lookup_rejects_unknown_phase():
    result = _solve()
    with pytest.raises(KeyError):
        result.relative_permeability("solid")


# ---------------------------------------------------------------------------
# solve_trickle_bed: input validation
# ---------------------------------------------------------------------------

def test_rejects_non_system_input():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_trickle_bed(
            "not a system", "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_systems_with_more_than_two_phases():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.2)
    oil = Phase("oil", mesh, density=850.0, viscosity=5e-3, volume_fraction=0.3)
    water = Phase("water", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    system = EulerianMultiphaseSystem(mesh, [gas, oil, water])
    with pytest.raises(ValueError):
        solve_trickle_bed(
            system, "water", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_unknown_liquid_phase_name():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_trickle_bed(
            system, "solid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [True, "1e-3", None, 0.0, -1.0])
def test_rejects_invalid_particle_diameter(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        solve_trickle_bed(
            system, "liquid", bad_value, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_bad_bed_properties_type():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(TypeError):
        solve_trickle_bed(
            system, "liquid", 1e-3, "not bed properties", _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_bad_capillary_pressure_model_type():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(TypeError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), "not a model", _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_bad_relative_permeability_model_type():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(TypeError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), "not a model",
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [True, "0.01", None, 0.0, -1.0])
def test_rejects_invalid_dt(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, bad_value, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_total_time_not_a_multiple_of_dt():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.025, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [0.0, 1.5, -0.2])
def test_rejects_invalid_velocity_relaxation(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(),
            velocity_relaxation=bad_value, max_outer_iterations=1,
        )


def test_rejects_missing_u_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    incomplete = [FixedValueBC("left", 0.0), FixedValueBC("right", 0.0), FixedValueBC("top", 0.0)]
    with pytest.raises(ValueError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, incomplete, _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_non_dirichlet_v_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    boundaries = [
        FixedValueBC("left", 0.0), FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0), ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), boundaries, max_outer_iterations=1,
        )


def test_rejects_non_uniform_mesh_spacing():
    from src.cfd.mesh import Mesh

    xs = [0.0, 1.0, 3.0]
    ys = [0.0, 1.0, 2.0]
    cell_centers = np.array([[x, y] for y in ys for x in xs])
    mesh = Mesh(
        cell_centers=cell_centers,
        face_centers=cell_centers[:1],
        face_areas=np.ones(1),
        cell_volumes=np.ones(len(cell_centers)),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.3)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.7)
    system = EulerianMultiphaseSystem(mesh, [gas, liquid])
    with pytest.raises(ValueError):
        solve_trickle_bed(
            system, "liquid", 1e-3, _bed_properties(), _capillary_model(mesh), _kr_model(mesh),
            0.07, 0.5, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_leverett_capillary_model_runs_without_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = _solve(mesh=mesh, capillary_model=LeverettJFunctionModel(mesh))
    assert np.all(np.isfinite(result.velocity("liquid").values))
    assert np.all(np.isfinite(result.velocity("gas").values))
