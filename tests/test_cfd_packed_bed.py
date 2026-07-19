import numpy as np
import pytest

from src.cfd import FixedValueBC, ScalarField, VectorField
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.eulerian_solver import assemble_phase_momentum_system
from src.cfd.multiphase import Phase
from src.cfd.packed_bed import (
    ErgunResistanceModel,
    PackedBedProperties,
    assemble_packed_bed_phase_momentum_system,
    packed_bed_momentum_equation,
)


def _u_boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


def _manual_coefficients(dp, eps, mu, rho):
    a = 150.0 * mu * (1.0 - eps) ** 2 / (eps ** 3 * dp ** 2)
    b = 1.75 * rho * (1.0 - eps) / (eps ** 3 * dp)
    return a, b


# ---------------------------------------------------------------------------
# packed_bed_momentum_equation
# ---------------------------------------------------------------------------

def test_packed_bed_momentum_equation_contains_phase_name_and_resistance_term():
    equation = packed_bed_momentum_equation("liquid")
    text = str(equation)
    assert "U_liquid" in text
    assert "ergun_resistance" in text
    assert "alpha_liquid" in text


def test_packed_bed_momentum_equation_rejects_empty_phase_name():
    with pytest.raises(ValueError):
        packed_bed_momentum_equation("")


def test_packed_bed_momentum_equation_rejects_non_string_phase_name():
    with pytest.raises(ValueError):
        packed_bed_momentum_equation("   ")


# ---------------------------------------------------------------------------
# PackedBedProperties: construction and coefficients
# ---------------------------------------------------------------------------

def test_viscous_and_inertial_coefficients_match_manual_ergun_formula():
    dp, eps, mu, rho = 1e-3, 0.4, 1e-3, 1000.0
    props = PackedBedProperties(dp, eps, mu, rho)
    expected_a, expected_b = _manual_coefficients(dp, eps, mu, rho)
    assert np.isclose(props.viscous_coefficient, expected_a)
    assert np.isclose(props.inertial_coefficient, expected_b)


@pytest.mark.parametrize("bad_value", [True, "1e-3", None, 0.0, -1.0])
def test_rejects_invalid_particle_diameter(bad_value):
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        PackedBedProperties(bad_value, 0.4, 1e-3, 1000.0)


@pytest.mark.parametrize("bad_value", [True, "0.4", None, 0.0, 1.0, -0.1, 1.1])
def test_rejects_invalid_porosity(bad_value):
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        PackedBedProperties(1e-3, bad_value, 1e-3, 1000.0)


@pytest.mark.parametrize("bad_value", [True, "1e-3", None, 0.0, -1.0])
def test_rejects_invalid_fluid_viscosity(bad_value):
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        PackedBedProperties(1e-3, 0.4, bad_value, 1000.0)


@pytest.mark.parametrize("bad_value", [True, "1000.0", None, 0.0, -1.0])
def test_rejects_invalid_fluid_density(bad_value):
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        PackedBedProperties(1e-3, 0.4, 1e-3, bad_value)


@pytest.mark.parametrize("bad_value", [True, "0.1", -1.0])
def test_rejects_invalid_superficial_velocity(bad_value):
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0, superficial_velocity=bad_value)


def test_superficial_velocity_of_zero_is_accepted():
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0, superficial_velocity=0.0)
    assert props.superficial_velocity == 0.0


def test_repr_contains_key_fields():
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    text = repr(props)
    assert "particle_diameter=0.001" in text
    assert "porosity=0.4" in text


# ---------------------------------------------------------------------------
# PackedBedProperties: pressure_gradient / pressure_drop
# ---------------------------------------------------------------------------

def test_pressure_gradient_matches_manual_ergun_formula():
    dp, eps, mu, rho, velocity = 1e-3, 0.4, 1e-3, 1000.0, 0.05
    props = PackedBedProperties(dp, eps, mu, rho)
    a, b = _manual_coefficients(dp, eps, mu, rho)
    expected = a * velocity + b * velocity ** 2
    assert np.isclose(props.pressure_gradient(velocity), expected)


def test_pressure_gradient_uses_stored_superficial_velocity_by_default():
    dp, eps, mu, rho, velocity = 1e-3, 0.4, 1e-3, 1000.0, 0.05
    props = PackedBedProperties(dp, eps, mu, rho, superficial_velocity=velocity)
    a, b = _manual_coefficients(dp, eps, mu, rho)
    expected = a * velocity + b * velocity ** 2
    assert np.isclose(props.pressure_gradient(), expected)


def test_pressure_gradient_raises_without_velocity_or_stored_default():
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    with pytest.raises(ValueError):
        props.pressure_gradient()


def test_pressure_drop_scales_linearly_with_bed_length():
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0, superficial_velocity=0.05)
    gradient = props.pressure_gradient()
    assert np.isclose(props.pressure_drop(2.0), gradient * 2.0)


def test_pressure_drop_rejects_non_positive_bed_length():
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0, superficial_velocity=0.05)
    with pytest.raises(ValueError):
        props.pressure_drop(0.0)


# ---------------------------------------------------------------------------
# ErgunResistanceModel: construction
# ---------------------------------------------------------------------------

def test_ergun_model_rejects_non_mesh():
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    with pytest.raises(TypeError):
        ErgunResistanceModel("not a mesh", props)


def test_ergun_model_rejects_non_properties():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        ErgunResistanceModel(mesh, "not properties")


# ---------------------------------------------------------------------------
# ErgunResistanceModel: resistance_coefficient
# ---------------------------------------------------------------------------

def test_resistance_coefficient_matches_manual_computation():
    dp, eps, mu, rho = 1e-3, 0.4, 1e-3, 1000.0
    mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(dp, eps, mu, rho)
    model = ErgunResistanceModel(mesh, props)

    velocity = VectorField(mesh, np.full((mesh.n_cells, 2), [0.03, 0.04]))
    result = model.resistance_coefficient(velocity)

    a, b = _manual_coefficients(dp, eps, mu, rho)
    expected = a + b * 0.05  # |[0.03, 0.04]| == 0.05
    assert np.allclose(result.values, expected)


def test_resistance_coefficient_at_zero_velocity_equals_viscous_coefficient_only():
    mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    model = ErgunResistanceModel(mesh, props)

    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    result = model.resistance_coefficient(velocity)

    assert np.all(np.isfinite(result.values))
    assert np.allclose(result.values, props.viscous_coefficient)


def test_resistance_coefficient_rejects_non_vector_field():
    mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    model = ErgunResistanceModel(mesh, props)
    with pytest.raises(TypeError):
        model.resistance_coefficient("not a field")


def test_resistance_coefficient_rejects_mismatched_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    model = ErgunResistanceModel(mesh, props)
    velocity = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        model.resistance_coefficient(velocity)


# ---------------------------------------------------------------------------
# ErgunResistanceModel: momentum_source
# ---------------------------------------------------------------------------

def test_momentum_source_opposes_flow_direction():
    mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    model = ErgunResistanceModel(mesh, props)

    velocity = VectorField(mesh, np.full((mesh.n_cells, 2), [0.1, 0.0]))
    source = model.momentum_source(velocity)

    assert np.all(source.values[:, 0] < 0.0)
    assert np.allclose(source.values[:, 1], 0.0)


def test_momentum_source_magnitude_matches_coefficient_times_speed():
    mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    model = ErgunResistanceModel(mesh, props)

    velocity = VectorField(mesh, np.full((mesh.n_cells, 2), [0.03, 0.04]))
    coefficient = model.resistance_coefficient(velocity)
    source = model.momentum_source(velocity)

    magnitude = np.sqrt(np.sum(source.values ** 2, axis=1))
    assert np.allclose(magnitude, coefficient.values * 0.05)


def test_momentum_source_is_zero_at_zero_velocity():
    mesh = build_structured_mesh(nx=3, ny=3)
    props = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    model = ErgunResistanceModel(mesh, props)

    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    source = model.momentum_source(velocity)
    assert np.allclose(source.values, 0.0)


# ---------------------------------------------------------------------------
# assemble_packed_bed_phase_momentum_system
# ---------------------------------------------------------------------------

def _phase_momentum_inputs(mesh):
    phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.7)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    other_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    drag = ScalarField(mesh, np.zeros(mesh.n_cells))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    return phase, velocity, other_velocity, drag, pressure


def test_assemble_packed_bed_system_returns_correctly_sized_system():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)

    system = assemble_packed_bed_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, bed, _u_boundaries(), "u"
    )
    assert system.size == mesh.n_cells


def test_assemble_packed_bed_system_diagonal_exceeds_plain_momentum_system():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)

    plain_system = assemble_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, _u_boundaries(), "u"
    )
    packed_bed_system = assemble_packed_bed_phase_momentum_system(
        mesh, phase, velocity, other_velocity, drag, pressure, bed, _u_boundaries(), "u"
    )

    interior_cell = mesh.n_cells // 2
    assert packed_bed_system.matrix.get(interior_cell, interior_cell) > plain_system.matrix.get(
        interior_cell, interior_cell
    )


def test_assemble_packed_bed_system_rejects_non_packed_bed_properties():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    with pytest.raises(TypeError):
        assemble_packed_bed_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, "not bed properties", _u_boundaries(), "u"
        )


def test_assemble_packed_bed_system_rejects_wrong_component():
    mesh = build_structured_mesh(nx=5, ny=5)
    phase, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    with pytest.raises(ValueError):
        assemble_packed_bed_phase_momentum_system(
            mesh, phase, velocity, other_velocity, drag, pressure, bed, _u_boundaries(), "w"
        )


def test_assemble_packed_bed_system_rejects_non_phase():
    mesh = build_structured_mesh(nx=5, ny=5)
    _, velocity, other_velocity, drag, pressure = _phase_momentum_inputs(mesh)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    with pytest.raises(TypeError):
        assemble_packed_bed_phase_momentum_system(
            mesh, "not a phase", velocity, other_velocity, drag, pressure, bed, _u_boundaries(), "u"
        )
