import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ScalarField, VectorField
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.multiphase import (
    EulerianMultiphaseSystem,
    Phase,
    phase_continuity_equation,
    volume_fraction_closure_equation,
)
from src.cfd.navier_stokes import NavierStokesResult


def _boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


# ---------------------------------------------------------------------------
# Phase construction
# ---------------------------------------------------------------------------

def test_phase_with_scalar_volume_fraction_broadcasts_to_every_cell():
    mesh = build_structured_mesh(nx=3, ny=3)
    phase = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.3)
    assert phase.name == "gas"
    assert phase.density == 1.2
    assert phase.viscosity == 1.8e-5
    assert np.allclose(phase.volume_fraction.values, 0.3)
    assert phase.volume_fraction.mesh is mesh


def test_phase_defaults_velocity_and_pressure_to_zero():
    mesh = build_structured_mesh(nx=3, ny=3)
    phase = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=1.0)
    assert phase.velocity.values.shape == (mesh.n_cells, 2)
    assert np.allclose(phase.velocity.values, 0.0)
    assert phase.pressure.values.shape == (mesh.n_cells,)
    assert np.allclose(phase.pressure.values, 0.0)


def test_phase_accepts_array_like_volume_fraction():
    mesh = build_structured_mesh(nx=3, ny=3)
    values = np.linspace(0.0, 1.0, mesh.n_cells)
    phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=values)
    assert np.allclose(phase.volume_fraction.values, values)


def test_phase_accepts_scalar_field_volume_fraction():
    mesh = build_structured_mesh(nx=3, ny=3)
    field = ScalarField(mesh, np.full(mesh.n_cells, 0.6))
    phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=field)
    assert np.allclose(phase.volume_fraction.values, 0.6)
    # Stored volume fraction must be an independent copy.
    field.values[0] = 0.0
    assert phase.volume_fraction.values[0] == 0.6


def test_phase_accepts_explicit_velocity_and_pressure_fields():
    mesh = build_structured_mesh(nx=3, ny=3)
    velocity = VectorField(mesh, np.full((mesh.n_cells, 2), 2.0))
    pressure = ScalarField(mesh, np.full(mesh.n_cells, 5.0))
    phase = Phase(
        "gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=1.0,
        velocity=velocity, pressure=pressure,
    )
    assert np.allclose(phase.velocity.values, 2.0)
    assert np.allclose(phase.pressure.values, 5.0)


def test_phase_rejects_non_string_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        Phase(123, mesh, density=1.0, viscosity=1.0, volume_fraction=1.0)


def test_phase_rejects_empty_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(ValueError):
        Phase("   ", mesh, density=1.0, viscosity=1.0, volume_fraction=1.0)


def test_phase_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        Phase("gas", "not a mesh", density=1.0, viscosity=1.0, volume_fraction=1.0)


@pytest.mark.parametrize("bad_density", [True, "1.0", None, 0.0, -1.0])
def test_phase_rejects_invalid_density(bad_density):
    mesh = build_structured_mesh(nx=3, ny=3)
    error = TypeError if not isinstance(bad_density, (int, float)) or isinstance(bad_density, bool) else ValueError
    with pytest.raises(error):
        Phase("gas", mesh, density=bad_density, viscosity=1.0, volume_fraction=1.0)


@pytest.mark.parametrize("bad_viscosity", [True, "1.0", None, 0.0, -1.0])
def test_phase_rejects_invalid_viscosity(bad_viscosity):
    mesh = build_structured_mesh(nx=3, ny=3)
    error = TypeError if not isinstance(bad_viscosity, (int, float)) or isinstance(bad_viscosity, bool) else ValueError
    with pytest.raises(error):
        Phase("gas", mesh, density=1.0, viscosity=bad_viscosity, volume_fraction=1.0)


def test_phase_rejects_volume_fraction_below_zero():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(ValueError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=-0.1)


def test_phase_rejects_volume_fraction_above_one():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(ValueError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=1.1)


def test_phase_rejects_wrong_type_volume_fraction():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction="0.5")


def test_phase_rejects_volume_fraction_field_on_different_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    field = ScalarField(other_mesh, np.full(other_mesh.n_cells, 0.5))
    with pytest.raises(ValueError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=field)


def test_phase_rejects_velocity_on_different_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    velocity = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=1.0, velocity=velocity)


def test_phase_rejects_pressure_on_different_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    pressure = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=1.0, pressure=pressure)


def test_phase_rejects_wrong_type_velocity():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=1.0, velocity="not a field")


def test_phase_rejects_wrong_type_pressure():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=1.0, pressure="not a field")


def test_phase_repr_contains_name_and_density():
    mesh = build_structured_mesh(nx=3, ny=3)
    phase = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=1.0)
    text = repr(phase)
    assert "gas" in text
    assert "1.2" in text


# ---------------------------------------------------------------------------
# EulerianMultiphaseSystem construction and validation
# ---------------------------------------------------------------------------

def _two_phase_system(mesh, alpha_gas=0.3):
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=alpha_gas)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=1.0 - alpha_gas)
    return EulerianMultiphaseSystem(mesh, [gas, liquid])


def test_two_phase_system_constructs_successfully():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh)
    assert system.n_phases == 2
    assert system.phase_names == ["gas", "liquid"]


def test_three_phase_system_with_matching_fractions_constructs_successfully():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.2)
    oil = Phase("oil", mesh, density=850.0, viscosity=5e-3, volume_fraction=0.3)
    water = Phase("water", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    system = EulerianMultiphaseSystem(mesh, [gas, oil, water])
    assert system.n_phases == 3
    assert np.allclose(system.volume_fraction_sum().values, 1.0)


def test_system_rejects_non_mesh_input():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.5)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    with pytest.raises(TypeError):
        EulerianMultiphaseSystem("not a mesh", [gas, liquid])


def test_system_rejects_non_sequence_phases():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        EulerianMultiphaseSystem(mesh, "not a sequence")


def test_system_rejects_non_phase_entries():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.5)
    with pytest.raises(TypeError):
        EulerianMultiphaseSystem(mesh, [gas, "not a phase"])


def test_system_rejects_single_phase():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=1.0)
    with pytest.raises(ValueError):
        EulerianMultiphaseSystem(mesh, [gas])


def test_system_rejects_duplicate_phase_names():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas_a = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.5)
    gas_b = Phase("gas", mesh, density=1.3, viscosity=1.9e-5, volume_fraction=0.5)
    with pytest.raises(ValueError):
        EulerianMultiphaseSystem(mesh, [gas_a, gas_b])


def test_system_rejects_phase_attached_to_different_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.5)
    liquid = Phase("liquid", other_mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    with pytest.raises(ValueError):
        EulerianMultiphaseSystem(mesh, [gas, liquid])


def test_system_rejects_volume_fractions_not_summing_to_one():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.3)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.3)
    with pytest.raises(ValueError):
        EulerianMultiphaseSystem(mesh, [gas, liquid])


def test_system_accepts_fractions_within_tolerance():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.3 + 1e-9)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.7)
    system = EulerianMultiphaseSystem(mesh, [gas, liquid], volume_fraction_tolerance=1e-6)
    assert system.n_phases == 2


@pytest.mark.parametrize("bad_tolerance", [True, "1e-6", None, 0.0, -1.0])
def test_system_rejects_invalid_volume_fraction_tolerance(bad_tolerance):
    mesh = build_structured_mesh(nx=3, ny=3)
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.5)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.5)
    error = TypeError if not isinstance(bad_tolerance, (int, float)) or isinstance(bad_tolerance, bool) else ValueError
    with pytest.raises(error):
        EulerianMultiphaseSystem(mesh, [gas, liquid], volume_fraction_tolerance=bad_tolerance)


def test_system_repr_contains_phase_names():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh)
    text = repr(system)
    assert "gas" in text
    assert "liquid" in text


# ---------------------------------------------------------------------------
# get_phase
# ---------------------------------------------------------------------------

def test_get_phase_returns_matching_phase():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh)
    assert system.get_phase("gas").name == "gas"
    assert system.get_phase("liquid").name == "liquid"


def test_get_phase_rejects_non_string_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh)
    with pytest.raises(TypeError):
        system.get_phase(123)


def test_get_phase_raises_key_error_for_unknown_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh)
    with pytest.raises(KeyError):
        system.get_phase("solid")


# ---------------------------------------------------------------------------
# Mixture properties
# ---------------------------------------------------------------------------

def test_mixture_density_field_is_volume_fraction_weighted():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh, alpha_gas=0.4)
    expected = 0.4 * 1.2 + 0.6 * 1000.0
    assert np.allclose(system.mixture_density_field().values, expected)
    assert np.isclose(system.mixture_density(), expected)


def test_mixture_viscosity_field_is_volume_fraction_weighted():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh, alpha_gas=0.4)
    expected = 0.4 * 1.8e-5 + 0.6 * 1e-3
    assert np.allclose(system.mixture_viscosity_field().values, expected)
    assert np.isclose(system.mixture_viscosity(), expected)


def test_mixture_velocity_is_mass_weighted_average():
    mesh = build_structured_mesh(nx=3, ny=3)
    gas_velocity = VectorField(mesh, np.full((mesh.n_cells, 2), [2.0, 0.0]))
    liquid_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    gas = Phase("gas", mesh, density=1.0, viscosity=1.0, volume_fraction=0.5, velocity=gas_velocity)
    liquid = Phase("liquid", mesh, density=3.0, viscosity=1.0, volume_fraction=0.5, velocity=liquid_velocity)
    system = EulerianMultiphaseSystem(mesh, [gas, liquid])

    # weight_gas = 0.5 * 1.0 = 0.5, weight_liquid = 0.5 * 3.0 = 1.5
    # mixture_u = (0.5 * 2.0 + 1.5 * 0.0) / (0.5 + 1.5) = 0.5
    mixture_velocity = system.mixture_velocity()
    assert np.allclose(mixture_velocity.values[:, 0], 0.5)
    assert np.allclose(mixture_velocity.values[:, 1], 0.0)


def test_volume_fraction_sum_equals_one_for_valid_system():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh, alpha_gas=0.25)
    assert np.allclose(system.volume_fraction_sum().values, 1.0)


def test_revalidate_volume_fractions_raises_after_mutation():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh, alpha_gas=0.3)
    system.get_phase("gas").volume_fraction.values[:] = 0.9
    with pytest.raises(ValueError):
        system.revalidate_volume_fractions()


def test_revalidate_volume_fractions_passes_when_still_consistent():
    mesh = build_structured_mesh(nx=3, ny=3)
    system = _two_phase_system(mesh, alpha_gas=0.3)
    system.revalidate_volume_fractions()  # should not raise


# ---------------------------------------------------------------------------
# solve_mixture_flow
# ---------------------------------------------------------------------------

def test_solve_mixture_flow_returns_navier_stokes_result():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _two_phase_system(mesh, alpha_gas=0.3)
    result = system.solve_mixture_flow(
        _boundaries(top=1.0), _boundaries(), max_outer_iterations=10,
    )
    assert isinstance(result, NavierStokesResult)
    assert result.velocity.values.shape == (mesh.n_cells, 2)
    assert result.pressure.values.shape == (mesh.n_cells,)
    assert np.all(np.isfinite(result.velocity.values))
    assert np.isclose(result.density, system.mixture_density())
    assert np.isclose(result.dynamic_viscosity, system.mixture_viscosity())


def test_solve_mixture_flow_zero_boundaries_gives_zero_velocity():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = _two_phase_system(mesh, alpha_gas=0.3)
    result = system.solve_mixture_flow(
        _boundaries(), _boundaries(), max_outer_iterations=10, outer_tolerance=1e-8,
    )
    assert result.converged is True
    assert np.allclose(result.velocity.values, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Equation helpers
# ---------------------------------------------------------------------------

def test_phase_continuity_equation_contains_phase_name():
    equation = phase_continuity_equation("gas")
    text = str(equation)
    assert "alpha_gas" in text
    assert "divergence" in text


def test_phase_continuity_equation_rejects_empty_name():
    with pytest.raises(ValueError):
        phase_continuity_equation("")


def test_volume_fraction_closure_equation_lists_every_phase():
    equation = volume_fraction_closure_equation(["gas", "liquid", "solid"])
    text = str(equation)
    assert "alpha_gas" in text
    assert "alpha_liquid" in text
    assert "alpha_solid" in text
    assert text.endswith("= 1")


def test_volume_fraction_closure_equation_rejects_empty_sequence():
    with pytest.raises(ValueError):
        volume_fraction_closure_equation([])
