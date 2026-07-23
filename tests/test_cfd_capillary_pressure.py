import numpy as np
import pytest

from src.cfd import ScalarField
from src.cfd.capillary_pressure import (
    AttouFerschneiderModel,
    CapillaryPressureModel,
    ConstantCapillaryPressureModel,
    LeverettJFunctionModel,
    capillary_pressure_closure_equation,
    capillary_pressure_from_phase,
)
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.multiphase import Phase
from src.cfd.packed_bed import PackedBedProperties


def _saturation_field(mesh, value):
    return ScalarField(mesh, np.full(mesh.n_cells, value))


def _leverett_manual(sl, eps, dp, sigma, theta):
    k = (eps ** 3 * dp ** 2) / (150.0 * (1.0 - eps) ** 2)
    one_minus_s = 1.0 - sl
    j = 1.417 * one_minus_s - 2.120 * one_minus_s ** 2 + 1.263 * one_minus_s ** 3
    return sigma * np.cos(theta) * np.sqrt(eps / k) * j


def _attou_ferschneider_manual(sl, eps, dp, sigma, theta):
    structural_factor = np.sqrt((1.0 - eps) / eps)
    scale = sigma * np.cos(theta) / dp
    return scale * structural_factor * (1.0 - sl) / sl


# ---------------------------------------------------------------------------
# capillary_pressure_closure_equation
# ---------------------------------------------------------------------------

def test_capillary_pressure_closure_equation_contains_phase_name():
    equation = capillary_pressure_closure_equation("liquid")
    text = str(equation)
    assert "p_c_liquid" in text
    assert "S_liquid" in text


def test_capillary_pressure_closure_equation_rejects_empty_phase_name():
    with pytest.raises(ValueError):
        capillary_pressure_closure_equation("")


def test_capillary_pressure_closure_equation_rejects_whitespace_phase_name():
    with pytest.raises(ValueError):
        capillary_pressure_closure_equation("   ")


# ---------------------------------------------------------------------------
# CapillaryPressureModel base class
# ---------------------------------------------------------------------------

def test_capillary_pressure_model_cannot_be_instantiated_directly():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        CapillaryPressureModel(mesh)


def test_capillary_pressure_model_rejects_non_mesh():
    with pytest.raises(TypeError):
        ConstantCapillaryPressureModel("not a mesh", 100.0)


# ---------------------------------------------------------------------------
# Shared input validation (exercised through ConstantCapillaryPressureModel)
# ---------------------------------------------------------------------------

def test_rejects_non_scalar_field_saturation():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    with pytest.raises(TypeError):
        model.capillary_pressure("not a field", 0.4, 1e-3, 0.07, 0.0)


def test_rejects_saturation_field_on_mismatched_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(other_mesh, 0.5)
    with pytest.raises(ValueError):
        model.capillary_pressure(saturation, 0.4, 1e-3, 0.07, 0.0)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_rejects_saturation_values_outside_unit_range(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, bad_value)
    with pytest.raises(ValueError):
        model.capillary_pressure(saturation, 0.4, 1e-3, 0.07, 0.0)


@pytest.mark.parametrize("bad_value", [True, "0.4", None, 0.0, 1.0, -0.1, 1.1])
def test_rejects_invalid_porosity(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, 0.5)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        model.capillary_pressure(saturation, bad_value, 1e-3, 0.07, 0.0)


@pytest.mark.parametrize("bad_value", [True, "1e-3", None, 0.0, -1.0])
def test_rejects_invalid_particle_diameter(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, 0.5)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        model.capillary_pressure(saturation, 0.4, bad_value, 0.07, 0.0)


@pytest.mark.parametrize("bad_value", [True, "0.07", None, -0.01])
def test_rejects_invalid_surface_tension(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, 0.5)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        model.capillary_pressure(saturation, 0.4, 1e-3, bad_value, 0.0)


def test_surface_tension_of_zero_is_accepted():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, 0.5)
    result = model.capillary_pressure(saturation, 0.4, 1e-3, 0.0, 0.0)
    assert np.allclose(result.values, 100.0)


@pytest.mark.parametrize("bad_value", [True, "0.0", None, -0.1, np.pi + 0.1])
def test_rejects_invalid_contact_angle(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, 0.5)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        model.capillary_pressure(saturation, 0.4, 1e-3, 0.07, bad_value)


@pytest.mark.parametrize("angle", [0.0, np.pi])
def test_contact_angle_boundary_values_are_accepted(angle):
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 100.0)
    saturation = _saturation_field(mesh, 0.5)
    result = model.capillary_pressure(saturation, 0.4, 1e-3, 0.07, angle)
    assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# ConstantCapillaryPressureModel
# ---------------------------------------------------------------------------

def test_constant_model_returns_same_value_everywhere():
    mesh = build_structured_mesh(nx=4, ny=4)
    model = ConstantCapillaryPressureModel(mesh, 250.0)
    saturation = _saturation_field(mesh, 0.3)
    result = model.capillary_pressure(saturation, 0.4, 1e-3, 0.07, 0.5)
    assert np.allclose(result.values, 250.0)


def test_constant_model_ignores_saturation_variation():
    mesh = build_structured_mesh(nx=4, ny=4)
    model = ConstantCapillaryPressureModel(mesh, 250.0)
    varying_saturation = ScalarField(mesh, np.linspace(0.0, 1.0, mesh.n_cells))
    result = model.capillary_pressure(varying_saturation, 0.4, 1e-3, 0.07, 0.5)
    assert np.allclose(result.values, 250.0)


@pytest.mark.parametrize("bad_value", [True, "100.0", None, -1.0])
def test_constant_model_rejects_invalid_construction_value(bad_value):
    mesh = build_structured_mesh(nx=3, ny=3)
    error = TypeError if not isinstance(bad_value, (int, float)) or isinstance(bad_value, bool) else ValueError
    with pytest.raises(error):
        ConstantCapillaryPressureModel(mesh, bad_value)


def test_constant_model_accepts_zero():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 0.0)
    assert model.capillary_pressure_value == 0.0


def test_constant_model_repr_contains_value():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 250.0)
    assert "250.0" in repr(model)


# ---------------------------------------------------------------------------
# LeverettJFunctionModel
# ---------------------------------------------------------------------------

def test_leverett_model_matches_manual_computation():
    mesh = build_structured_mesh(nx=3, ny=3)
    sl, eps, dp, sigma, theta = 0.6, 0.4, 1e-3, 0.072, 0.3
    model = LeverettJFunctionModel(mesh)
    saturation = _saturation_field(mesh, sl)

    result = model.capillary_pressure(saturation, eps, dp, sigma, theta)
    expected = _leverett_manual(sl, eps, dp, sigma, theta)
    assert np.allclose(result.values, expected)


def test_leverett_model_varies_with_saturation():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = LeverettJFunctionModel(mesh)

    low_saturation = _saturation_field(mesh, 0.1)
    high_saturation = _saturation_field(mesh, 0.9)

    low_result = model.capillary_pressure(low_saturation, 0.4, 1e-3, 0.072, 0.0)
    high_result = model.capillary_pressure(high_saturation, 0.4, 1e-3, 0.072, 0.0)

    assert not np.allclose(low_result.values, high_result.values)


def test_leverett_model_zero_surface_tension_gives_zero_pressure():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = LeverettJFunctionModel(mesh)
    saturation = _saturation_field(mesh, 0.5)
    result = model.capillary_pressure(saturation, 0.4, 1e-3, 0.0, 0.0)
    assert np.allclose(result.values, 0.0)


def test_leverett_model_repr_uses_class_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = LeverettJFunctionModel(mesh)
    assert repr(model) == f"LeverettJFunctionModel(mesh=<{mesh.n_cells} cells>)"


# ---------------------------------------------------------------------------
# AttouFerschneiderModel
# ---------------------------------------------------------------------------

def test_attou_ferschneider_model_matches_manual_computation():
    mesh = build_structured_mesh(nx=3, ny=3)
    sl, eps, dp, sigma, theta = 0.6, 0.4, 1e-3, 0.072, 0.3
    model = AttouFerschneiderModel(mesh)
    saturation = _saturation_field(mesh, sl)

    result = model.capillary_pressure(saturation, eps, dp, sigma, theta)
    expected = _attou_ferschneider_manual(sl, eps, dp, sigma, theta)
    assert np.allclose(result.values, expected)


def test_attou_ferschneider_model_decreases_as_saturation_increases():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = AttouFerschneiderModel(mesh)

    low_saturation = _saturation_field(mesh, 0.2)
    high_saturation = _saturation_field(mesh, 0.8)

    low_result = model.capillary_pressure(low_saturation, 0.4, 1e-3, 0.072, 0.0)
    high_result = model.capillary_pressure(high_saturation, 0.4, 1e-3, 0.072, 0.0)

    assert np.all(low_result.values > high_result.values)


def test_attou_ferschneider_model_stays_finite_at_zero_saturation():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = AttouFerschneiderModel(mesh)
    saturation = _saturation_field(mesh, 0.0)
    result = model.capillary_pressure(saturation, 0.4, 1e-3, 0.072, 0.0)
    assert np.all(np.isfinite(result.values))
    assert np.all(result.values > 0.0)


def test_attou_ferschneider_model_approaches_zero_at_full_saturation():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = AttouFerschneiderModel(mesh)
    saturation = _saturation_field(mesh, 1.0)
    result = model.capillary_pressure(saturation, 0.4, 1e-3, 0.072, 0.0)
    assert np.allclose(result.values, 0.0)


# ---------------------------------------------------------------------------
# capillary_pressure_from_phase
# ---------------------------------------------------------------------------

def test_capillary_pressure_from_phase_uses_phase_volume_fraction_and_bed_properties():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 500.0)
    liquid_phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.6)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)

    result = capillary_pressure_from_phase(model, liquid_phase, bed, 0.072, 0.0)
    assert np.allclose(result.values, 500.0)


def test_capillary_pressure_from_phase_matches_direct_call_for_leverett():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = LeverettJFunctionModel(mesh)
    liquid_phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.6)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)

    via_helper = capillary_pressure_from_phase(model, liquid_phase, bed, 0.072, 0.3)
    direct = model.capillary_pressure(liquid_phase.volume_fraction, bed.porosity, bed.particle_diameter, 0.072, 0.3)
    assert np.allclose(via_helper.values, direct.values)


def test_capillary_pressure_from_phase_rejects_non_model():
    mesh = build_structured_mesh(nx=3, ny=3)
    liquid_phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.6)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    with pytest.raises(TypeError):
        capillary_pressure_from_phase("not a model", liquid_phase, bed, 0.072, 0.0)


def test_capillary_pressure_from_phase_rejects_non_phase():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 500.0)
    bed = PackedBedProperties(1e-3, 0.4, 1e-3, 1000.0)
    with pytest.raises(TypeError):
        capillary_pressure_from_phase(model, "not a phase", bed, 0.072, 0.0)


def test_capillary_pressure_from_phase_rejects_non_bed_properties():
    mesh = build_structured_mesh(nx=3, ny=3)
    model = ConstantCapillaryPressureModel(mesh, 500.0)
    liquid_phase = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.6)
    with pytest.raises(TypeError):
        capillary_pressure_from_phase(model, liquid_phase, "not bed properties", 0.072, 0.0)
