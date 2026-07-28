import numpy as np
import pytest

from src.cfd import FixedValueBC, ScalarField, VectorField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.force_accounting import ForceAccounting
from src.cfd.multiphase import EulerianMultiphaseSystem, Phase
from src.cfd.navier_stokes import NavierStokesResult
from src.cfd.packed_bed import PackedBedProperties
from src.cfd.transient_eulerian_solver import (
    TransientEulerianTwoFluidResult,
    assemble_transient_phase_momentum_system,
    solve_transient_eulerian_two_fluid,
    transient_phase_momentum_equation,
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


def _gas_liquid_system(mesh, alpha_gas=0.3):
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=alpha_gas)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=1.0 - alpha_gas)
    return EulerianMultiphaseSystem(mesh, [gas, liquid])


# ---------------------------------------------------------------------------
# transient_phase_momentum_equation
# ---------------------------------------------------------------------------

def test_transient_phase_momentum_equation_contains_ddt_and_both_phase_names():
    equation = transient_phase_momentum_equation("gas", "liquid")
    text = str(equation)
    assert "ddt(U_gas)" in text
    assert "U_liquid" in text
    assert "drag" in text
    assert "alpha_gas" in text


def test_transient_phase_momentum_equation_rejects_empty_phase_name():
    with pytest.raises(ValueError):
        transient_phase_momentum_equation("", "liquid")


def test_transient_phase_momentum_equation_rejects_empty_other_phase_name():
    with pytest.raises(ValueError):
        transient_phase_momentum_equation("gas", "")


# ---------------------------------------------------------------------------
# assemble_transient_phase_momentum_system
# ---------------------------------------------------------------------------

def _phase_momentum_inputs(mesh):
    phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.7)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    previous_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    other_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    drag = ScalarField(mesh, np.zeros(mesh.n_cells))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    return phase, velocity, previous_velocity, other_velocity, drag, pressure


def test_assemble_transient_phase_momentum_system_returns_correctly_sized_system():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, previous_velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    system = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.01
    )
    assert system.size == mesh.n_cells


def test_assemble_transient_phase_momentum_system_rejects_non_positive_dt():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, previous_velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(ValueError):
        assemble_transient_phase_momentum_system(
            mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.0
        )


def test_assemble_transient_phase_momentum_system_rejects_non_vector_previous_velocity():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, _, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(TypeError):
        assemble_transient_phase_momentum_system(
            mesh, phase, velocity, "not a field", other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.01
        )


def test_assemble_transient_phase_momentum_system_rejects_wrong_component():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, previous_velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(ValueError):
        assemble_transient_phase_momentum_system(
            mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "w", dt=0.01
        )


def test_assemble_transient_phase_momentum_system_rejects_bad_bed_properties():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, previous_velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(TypeError):
        assemble_transient_phase_momentum_system(
            mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u",
            dt=0.01, bed_properties="not bed properties",
        )


def test_assemble_transient_phase_momentum_system_diagonal_grows_as_dt_shrinks():
    # A smaller time step adds a larger 1/dt term to the diagonal, so the
    # smaller-dt system should have a strictly larger interior diagonal.
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, previous_velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)

    system_large_dt = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=1.0
    )
    system_small_dt = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.01
    )

    interior_cell = mesh.n_cells // 2
    assert system_small_dt.matrix.get(interior_cell, interior_cell) > system_large_dt.matrix.get(
        interior_cell, interior_cell
    )


def test_assemble_transient_phase_momentum_system_rhs_uses_previous_velocity():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, _, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    previous_velocity = VectorField(mesh, np.full((mesh.n_cells, 2), 2.0))

    system = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.5
    )
    interior_cell = mesh.n_cells // 2
    zero_previous_system = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, VectorField(mesh, np.zeros((mesh.n_cells, 2))), other_velocity, drag, pressure,
        _u_boundaries(), "u", dt=0.5,
    )
    assert system.rhs[interior_cell] > zero_previous_system.rhs[interior_cell]


def test_assemble_transient_phase_momentum_system_with_bed_properties_increases_diagonal():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, previous_velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    bed_properties = PackedBedProperties(
        particle_diameter=5e-3, porosity=0.4, fluid_viscosity=1e-3, fluid_density=1000.0
    )
    velocity = VectorField(mesh, np.full((mesh.n_cells, 2), [0.1, 0.0]))

    system_without_bed = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.5
    )
    system_with_bed = assemble_transient_phase_momentum_system(
        mesh, phase, velocity, previous_velocity, other_velocity, drag, pressure, _u_boundaries(), "u", dt=0.5,
        bed_properties=bed_properties,
    )

    interior_cell = mesh.n_cells // 2
    assert system_with_bed.matrix.get(interior_cell, interior_cell) > system_without_bed.matrix.get(
        interior_cell, interior_cell
    )


# ---------------------------------------------------------------------------
# solve_transient_eulerian_two_fluid: basic solving
# ---------------------------------------------------------------------------

def test_zero_boundary_conditions_give_near_zero_velocity_for_both_phases():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.02,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=10, mixture_max_outer_iterations=10,
    )
    assert result.converged is True
    assert np.allclose(result.velocity("gas").values, 0.0, atol=1e-6)
    assert np.allclose(result.velocity("liquid").values, 0.0, atol=1e-6)


def test_history_length_matches_number_of_time_steps():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.03,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5,
    )
    assert result.n_steps == 3
    assert len(result.time_points) == 4
    assert np.allclose(result.time_points, [0.0, 0.01, 0.02, 0.03])
    for name in ("gas", "liquid"):
        assert len(result.velocity_history[name]) == 4
        assert len(result.volume_fraction_history[name]) == 4
    assert len(result.pressure_history) == 4
    assert len(result.drag_coefficient_history) == 4
    assert len(result.iterations_per_step) == 4
    assert len(result.converged_per_step) == 4
    assert len(result.residual_history_per_step) == 4


def test_initial_history_entry_matches_system_initial_condition():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh, alpha_gas=0.25)
    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.01,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5,
    )
    assert np.allclose(result.volume_fraction("gas", step=0).values, 0.25)
    assert np.allclose(result.volume_fraction("liquid", step=0).values, 0.75)
    assert result.iterations_per_step[0] == 0
    assert result.converged_per_step[0] is True


def test_volume_fraction_history_is_constant_across_time():
    # Volume-fraction transport is not yet implemented (see module docstring),
    # so every stored step should carry the same (initial) volume fractions.
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh, alpha_gas=0.4)
    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.03,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5,
    )
    for field in result.volume_fraction_history["gas"]:
        assert np.allclose(field.values, 0.4)


def test_lid_driven_case_runs_without_error_and_stays_finite():
    mesh = build_structured_mesh(nx=6, ny=6)
    system = _gas_liquid_system(mesh, alpha_gas=0.2)
    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=2e-3, dt=0.01, total_time=0.02,
        u_boundary_conditions=_u_boundaries(top=1.0), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5,
    )
    for name in ("gas", "liquid"):
        velocity = result.velocity(name)
        assert velocity.values.shape == (mesh.n_cells, 2)
        assert np.all(np.isfinite(velocity.values))
    assert result.pressure().values.shape == (mesh.n_cells,)


def test_callable_boundary_condition_schedule_varies_with_time():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    seen_times = []

    def ramped_top(time):
        seen_times.append(time)
        return _u_boundaries(top=time)

    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.02,
        u_boundary_conditions=ramped_top, v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5,
    )
    assert result.n_steps == 2
    assert np.allclose(seen_times, [0.01, 0.02])


def test_callable_boundary_condition_schedule_rejects_incomplete_result():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)

    def bad_schedule(time):
        return _u_boundaries()[:3]

    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.01,
            u_boundary_conditions=bad_schedule, v_boundary_conditions=_v_boundaries(),
            max_outer_iterations=5, mixture_max_outer_iterations=5,
        )


def test_bed_properties_runs_without_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh, alpha_gas=0.1)
    bed_properties = PackedBedProperties(
        particle_diameter=5e-3, porosity=0.4, fluid_viscosity=1e-3, fluid_density=1000.0
    )
    result = solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.01,
        u_boundary_conditions=_u_boundaries(top=0.2), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5, bed_properties=bed_properties,
    )
    for name in ("gas", "liquid"):
        assert np.all(np.isfinite(result.velocity(name).values))
    assert result.bed_properties is bed_properties


# ---------------------------------------------------------------------------
# solve_transient_eulerian_two_fluid: input validation
# ---------------------------------------------------------------------------

def test_rejects_non_system_input():
    with pytest.raises(TypeError):
        solve_transient_eulerian_two_fluid(
            "not a system", "gas", 1e-3, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_systems_with_more_than_two_phases():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.2)
    oil = Phase("oil", mesh, density=850.0, viscosity=5e-3, volume_fraction=0.3)
    water = Phase("water", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    system = EulerianMultiphaseSystem(mesh, [gas, oil, water])
    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_unknown_dispersed_phase_name():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "solid", 1e-3, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [True, "1e-3", None, 0.0, -1.0])
def test_rejects_invalid_particle_diameter(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        solve_transient_eulerian_two_fluid(
            system, "gas", bad_value, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [True, "0.01", None, 0.0, -1.0])
def test_rejects_invalid_dt(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, bad_value, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_total_time_not_a_multiple_of_dt():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.025, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_total_time_smaller_than_dt():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.05, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [0.0, 1.5, -0.2])
def test_rejects_invalid_velocity_relaxation(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.01, _u_boundaries(), _v_boundaries(),
            velocity_relaxation=bad_value, max_outer_iterations=1,
        )


def test_rejects_missing_u_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    incomplete = [FixedValueBC("left", 0.0), FixedValueBC("right", 0.0), FixedValueBC("top", 0.0)]
    with pytest.raises(ValueError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.01, incomplete, _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_non_dirichlet_v_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    boundaries = [
        FixedValueBC("left", 0.0), FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0), ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.01, _u_boundaries(), boundaries, max_outer_iterations=1,
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
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_bad_bed_properties_type():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(TypeError):
        solve_transient_eulerian_two_fluid(
            system, "gas", 1e-3, 0.01, 0.01, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
            bed_properties="not bed properties",
        )


# ---------------------------------------------------------------------------
# TransientEulerianTwoFluidResult
# ---------------------------------------------------------------------------

def _small_result(bed_properties=None):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh, alpha_gas=0.3)
    return solve_transient_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3, dt=0.01, total_time=0.02,
        u_boundary_conditions=_u_boundaries(top=0.3), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5, mixture_max_outer_iterations=5, bed_properties=bed_properties,
    )


def test_result_phase_names():
    result = _small_result()
    assert set(result.phase_names) == {"gas", "liquid"}


def test_velocity_raises_key_error_for_unknown_phase():
    result = _small_result()
    with pytest.raises(KeyError):
        result.velocity("solid")


def test_velocity_raises_index_error_for_out_of_range_step():
    result = _small_result()
    with pytest.raises(IndexError):
        result.velocity("gas", step=100)


def test_volume_fraction_raises_key_error_for_unknown_phase():
    result = _small_result()
    with pytest.raises(KeyError):
        result.volume_fraction("solid")


def test_pressure_raises_index_error_for_out_of_range_step():
    result = _small_result()
    with pytest.raises(IndexError):
        result.pressure(step=100)


def test_result_repr_contains_key_fields():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = TransientEulerianTwoFluidResult(
        time_points=[0.0, 0.01],
        velocity_history={"gas": [VectorField(mesh, np.zeros((mesh.n_cells, 2)))] * 2},
        pressure_history=[ScalarField(mesh, np.zeros(mesh.n_cells))] * 2,
        volume_fraction_history={"gas": [ScalarField(mesh, np.zeros(mesh.n_cells))] * 2},
        drag_coefficient_history=[ScalarField(mesh, np.zeros(mesh.n_cells))] * 2,
        dt=0.01,
        total_time=0.01,
        n_steps=1,
        iterations_per_step=[0, 3],
        converged_per_step=[True, True],
        residual_history_per_step=[[], [{"gas_velocity": 0.0}]],
    )
    text = repr(result)
    assert "n_steps=1" in text
    assert "converged=True" in text


def test_build_force_accounting_returns_force_accounting_with_expected_contributions():
    result = _small_result()
    accounting = result.build_force_accounting("gas")
    assert isinstance(accounting, ForceAccounting)
    assert "pressure_gradient" in accounting
    assert "interphase_drag" in accounting
    assert "packed_bed_resistance" not in accounting


def test_build_force_accounting_registers_packed_bed_resistance_when_present():
    bed_properties = PackedBedProperties(
        particle_diameter=5e-3, porosity=0.4, fluid_viscosity=1e-3, fluid_density=1000.0
    )
    result = _small_result(bed_properties=bed_properties)
    accounting = result.build_force_accounting("liquid")
    assert "packed_bed_resistance" in accounting


def test_build_force_accounting_rejects_unknown_phase():
    result = _small_result()
    with pytest.raises(KeyError):
        result.build_force_accounting("solid")
