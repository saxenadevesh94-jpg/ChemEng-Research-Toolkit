import numpy as np
import pytest

from src.cfd import FixedValueBC, ScalarField, VectorField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.eulerian_solver import (
    EulerianTwoFluidResult,
    assemble_phase_momentum_system,
    interphase_drag_coefficient,
    phase_momentum_equation,
    relative_velocity_magnitude,
    schiller_naumann_drag_coefficient,
    solve_eulerian_two_fluid,
)
from src.cfd.multiphase import EulerianMultiphaseSystem, Phase
from src.cfd.navier_stokes import NavierStokesResult


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
# phase_momentum_equation
# ---------------------------------------------------------------------------

def test_phase_momentum_equation_contains_both_phase_names():
    equation = phase_momentum_equation("gas", "liquid")
    text = str(equation)
    assert "U_gas" in text
    assert "U_liquid" in text
    assert "drag" in text
    assert "alpha_gas" in text


def test_phase_momentum_equation_rejects_empty_phase_name():
    with pytest.raises(ValueError):
        phase_momentum_equation("", "liquid")


def test_phase_momentum_equation_rejects_empty_other_phase_name():
    with pytest.raises(ValueError):
        phase_momentum_equation("gas", "")


# ---------------------------------------------------------------------------
# schiller_naumann_drag_coefficient
# ---------------------------------------------------------------------------

def test_schiller_naumann_matches_laminar_formula_below_transition():
    reynolds = np.array([0.1, 1.0, 500.0, 1000.0])
    expected = (24.0 / reynolds) * (1.0 + 0.15 * reynolds ** 0.687)
    result = schiller_naumann_drag_coefficient(reynolds)
    assert np.allclose(result, expected)


def test_schiller_naumann_caps_at_newton_regime_above_transition():
    reynolds = np.array([1000.1, 5000.0])
    result = schiller_naumann_drag_coefficient(reynolds)
    assert np.allclose(result, 0.44)


def test_schiller_naumann_rejects_negative_reynolds():
    with pytest.raises(ValueError):
        schiller_naumann_drag_coefficient(np.array([-1.0]))


# ---------------------------------------------------------------------------
# relative_velocity_magnitude
# ---------------------------------------------------------------------------

def test_relative_velocity_magnitude_is_euclidean_norm_of_difference():
    mesh = build_structured_mesh(nx=3, ny=3)
    a = VectorField(mesh, np.full((mesh.n_cells, 2), [3.0, 0.0]))
    b = VectorField(mesh, np.full((mesh.n_cells, 2), [0.0, 4.0]))
    result = relative_velocity_magnitude(a, b)
    assert np.allclose(result.values, 5.0)


def test_relative_velocity_magnitude_rejects_mismatched_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    a = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    b = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        relative_velocity_magnitude(a, b)


def test_relative_velocity_magnitude_rejects_non_vector_field():
    mesh = build_structured_mesh(nx=3, ny=3)
    a = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    with pytest.raises(TypeError):
        relative_velocity_magnitude(a, "not a field")


# ---------------------------------------------------------------------------
# interphase_drag_coefficient
# ---------------------------------------------------------------------------

def test_interphase_drag_coefficient_matches_manual_computation():
    mesh = build_structured_mesh(nx=3, ny=3)
    alpha_d = ScalarField(mesh, np.full(mesh.n_cells, 0.3))
    relative_velocity = ScalarField(mesh, np.full(mesh.n_cells, 0.1))
    rho_c, mu_c, d_p = 1000.0, 1e-3, 1e-3

    result = interphase_drag_coefficient(mesh, rho_c, mu_c, alpha_d, relative_velocity, d_p)

    reynolds = rho_c * 0.1 * d_p / mu_c
    cd = (24.0 / reynolds) * (1.0 + 0.15 * reynolds ** 0.687)
    expected = 0.75 * cd * 0.3 * rho_c * 0.1 / d_p
    assert np.allclose(result.values, expected)


def test_interphase_drag_coefficient_is_finite_at_zero_relative_velocity():
    mesh = build_structured_mesh(nx=3, ny=3)
    alpha_d = ScalarField(mesh, np.full(mesh.n_cells, 0.3))
    relative_velocity = ScalarField(mesh, np.zeros(mesh.n_cells))
    result = interphase_drag_coefficient(mesh, 1000.0, 1e-3, alpha_d, relative_velocity, 1e-3)
    assert np.all(np.isfinite(result.values))
    assert np.all(result.values >= 0.0)


@pytest.mark.parametrize("bad_value", [True, "1.0", None, 0.0, -1.0])
def test_interphase_drag_coefficient_rejects_invalid_continuous_density(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    alpha_d = ScalarField(mesh, np.full(mesh.n_cells, 0.3))
    relative_velocity = ScalarField(mesh, np.full(mesh.n_cells, 0.1))
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        interphase_drag_coefficient(mesh, bad_value, 1e-3, alpha_d, relative_velocity, 1e-3)


@pytest.mark.parametrize("bad_value", [True, "1.0", None, 0.0, -1.0])
def test_interphase_drag_coefficient_rejects_invalid_particle_diameter(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    alpha_d = ScalarField(mesh, np.full(mesh.n_cells, 0.3))
    relative_velocity = ScalarField(mesh, np.full(mesh.n_cells, 0.1))
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        interphase_drag_coefficient(mesh, 1000.0, 1e-3, alpha_d, relative_velocity, bad_value)


def test_interphase_drag_coefficient_rejects_volume_fraction_on_different_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    alpha_d = ScalarField(other_mesh, np.full(other_mesh.n_cells, 0.3))
    relative_velocity = ScalarField(mesh, np.full(mesh.n_cells, 0.1))
    with pytest.raises(ValueError):
        interphase_drag_coefficient(mesh, 1000.0, 1e-3, alpha_d, relative_velocity, 1e-3)


def test_interphase_drag_coefficient_rejects_non_scalar_field_relative_velocity():
    mesh = build_structured_mesh(nx=3, ny=3)
    alpha_d = ScalarField(mesh, np.full(mesh.n_cells, 0.3))
    with pytest.raises(TypeError):
        interphase_drag_coefficient(mesh, 1000.0, 1e-3, alpha_d, "not a field", 1e-3)


# ---------------------------------------------------------------------------
# assemble_phase_momentum_system
# ---------------------------------------------------------------------------

def _phase_momentum_inputs(mesh):
    phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.7)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    other_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    drag = ScalarField(mesh, np.zeros(mesh.n_cells))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    return phase, velocity, other_velocity, drag, pressure


def test_assemble_phase_momentum_system_returns_correctly_sized_system():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    system = assemble_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, _u_boundaries(), "u"
    )
    assert system.size == mesh.n_cells


def test_assemble_phase_momentum_system_rejects_non_phase():
    mesh = build_structured_mesh(nx=5, ny=5)
    _, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(TypeError):
        assemble_phase_momentum_system(
            mesh, "not a phase", velocity, other_velocity, drag, pressure, _u_boundaries(), "u"
        )


def test_assemble_phase_momentum_system_rejects_wrong_component():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(ValueError):
        assemble_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, _u_boundaries(), "w"
        )


def test_assemble_phase_momentum_system_rejects_pressure_on_different_mesh():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, _ = _phase_momentum_inputs(mesh)
    bad_pressure = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        assemble_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, bad_pressure, _u_boundaries(), "u"
        )


def test_assemble_phase_momentum_system_rejects_velocity_with_one_component():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, _, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    bad_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 1)))
    with pytest.raises(ValueError):
        assemble_phase_momentum_system(
            mesh, phase, bad_velocity, other_velocity, drag, pressure, _u_boundaries(), "u"
        )


def test_assemble_phase_momentum_system_rejects_non_dirichlet_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    boundaries = [
        FixedValueBC("left", 0.0), FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0), ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        assemble_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, boundaries, "u"
        )


def test_assemble_phase_momentum_system_diagonal_grows_with_drag():
    # Adding drag should only ever add a non-negative amount to each
    # interior row's diagonal (it represents an implicit sink term).
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, _, pressure = _phase_momentum_inputs(mesh)
    zero_drag = ScalarField(mesh, np.zeros(mesh.n_cells))
    nonzero_drag = ScalarField(mesh, np.full(mesh.n_cells, 5.0))

    system_no_drag = assemble_phase_momentum_system(
        mesh, phase, velocity, other_velocity, zero_drag, pressure, _u_boundaries(), "u"
    )
    system_with_drag = assemble_phase_momentum_system(
        mesh, phase, velocity, other_velocity, nonzero_drag, pressure, _u_boundaries(), "u"
    )

    interior_cell = mesh.n_cells // 2
    assert system_with_drag.matrix.get(interior_cell, interior_cell) > system_no_drag.matrix.get(
        interior_cell, interior_cell
    )


# ---------------------------------------------------------------------------
# solve_eulerian_two_fluid: basic solving
# ---------------------------------------------------------------------------

def test_zero_boundary_conditions_give_near_zero_velocity_for_both_phases():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=15, outer_tolerance=1e-8,
    )
    assert result.converged is True
    assert np.allclose(result.velocity("gas").values, 0.0, atol=1e-6)
    assert np.allclose(result.velocity("liquid").values, 0.0, atol=1e-6)


def test_lid_driven_case_runs_without_error_and_stays_finite():
    mesh = build_structured_mesh(nx=6, ny=6)
    system = _gas_liquid_system(mesh, alpha_gas=0.2)
    result = solve_eulerian_two_fluid(
        system, "gas", particle_diameter=2e-3,
        u_boundary_conditions=_u_boundaries(top=1.0), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=10,
    )
    for name in ("gas", "liquid"):
        velocity = result.velocity(name)
        assert velocity.values.shape == (mesh.n_cells, 2)
        assert np.all(np.isfinite(velocity.values))
    assert result.pressure.values.shape == (mesh.n_cells,)
    assert result.drag_coefficient.values.shape == (mesh.n_cells,)
    assert np.all(np.isfinite(result.drag_coefficient.values))


def test_result_contains_both_phase_names():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_eulerian_two_fluid(
        system, "liquid", particle_diameter=1e-3,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5,
    )
    assert set(result.velocities) == {"gas", "liquid"}


def test_mixture_result_is_navier_stokes_result():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5,
    )
    assert isinstance(result.mixture_result, NavierStokesResult)


def test_residual_history_length_matches_iterations_run():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3,
        u_boundary_conditions=_u_boundaries(top=0.5), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=8,
    )
    assert len(result.residual_history) == result.iterations_run
    assert result.iterations_run <= 8


def test_velocity_raises_key_error_for_unknown_phase():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    result = solve_eulerian_two_fluid(
        system, "gas", particle_diameter=1e-3,
        u_boundary_conditions=_u_boundaries(), v_boundary_conditions=_v_boundaries(),
        max_outer_iterations=5,
    )
    with pytest.raises(KeyError):
        result.velocity("solid")


def test_eulerian_two_fluid_result_repr_contains_key_fields():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = EulerianTwoFluidResult(
        velocities={"gas": VectorField(mesh, np.zeros((mesh.n_cells, 2)))},
        pressure=ScalarField(mesh, np.zeros(mesh.n_cells)),
        drag_coefficient=ScalarField(mesh, np.zeros(mesh.n_cells)),
        mixture_result=None,
        iterations_run=4,
        converged=True,
        residual_history=[{"gas_velocity": 0.0}],
    )
    text = repr(result)
    assert "iterations_run=4" in text
    assert "converged=True" in text


# ---------------------------------------------------------------------------
# solve_eulerian_two_fluid: input validation
# ---------------------------------------------------------------------------

def test_rejects_non_system_input():
    with pytest.raises(TypeError):
        solve_eulerian_two_fluid(
            "not a system", "gas", 1e-3, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_systems_with_more_than_two_phases():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.2)
    oil = Phase("oil", mesh, density=850.0, viscosity=5e-3, volume_fraction=0.3)
    water = Phase("water", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    system = EulerianMultiphaseSystem(mesh, [gas, oil, water])
    with pytest.raises(ValueError):
        solve_eulerian_two_fluid(
            system, "gas", 1e-3, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_non_string_dispersed_phase():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(TypeError):
        solve_eulerian_two_fluid(
            system, 123, 1e-3, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_unknown_dispersed_phase_name():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_eulerian_two_fluid(
            system, "solid", 1e-3, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [True, "1e-3", None, 0.0, -1.0])
def test_rejects_invalid_particle_diameter(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        solve_eulerian_two_fluid(
            system, "gas", bad_value, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )


@pytest.mark.parametrize("bad_value", [0.0, 1.5, -0.2])
def test_rejects_invalid_velocity_relaxation(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_eulerian_two_fluid(
            system, "gas", 1e-3, _u_boundaries(), _v_boundaries(),
            velocity_relaxation=bad_value, max_outer_iterations=1,
        )


def test_rejects_non_positive_max_outer_iterations():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    with pytest.raises(ValueError):
        solve_eulerian_two_fluid(
            system, "gas", 1e-3, _u_boundaries(), _v_boundaries(), max_outer_iterations=0,
        )


def test_rejects_missing_u_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    incomplete = [FixedValueBC("left", 0.0), FixedValueBC("right", 0.0), FixedValueBC("top", 0.0)]
    with pytest.raises(ValueError):
        solve_eulerian_two_fluid(
            system, "gas", 1e-3, incomplete, _v_boundaries(), max_outer_iterations=1,
        )


def test_rejects_non_dirichlet_v_boundary():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _gas_liquid_system(mesh)
    boundaries = [
        FixedValueBC("left", 0.0), FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0), ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        solve_eulerian_two_fluid(
            system, "gas", 1e-3, _u_boundaries(), boundaries, max_outer_iterations=1,
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
        solve_eulerian_two_fluid(
            system, "gas", 1e-3, _u_boundaries(), _v_boundaries(), max_outer_iterations=1,
        )
