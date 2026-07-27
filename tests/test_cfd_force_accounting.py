import numpy as np
import pytest

from src.cfd.capillary_pressure import ConstantCapillaryPressureModel, capillary_pressure_from_phase
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.eulerian_solver import interphase_drag_coefficient, relative_velocity_magnitude
from src.cfd.field import VectorField
from src.cfd.force_accounting import ForceAccounting, ForceContribution
from src.cfd.multiphase import Phase
from src.cfd.operators import gradient
from src.cfd.packed_bed import ErgunResistanceModel, PackedBedProperties


def _vector_field(mesh, value):
    return VectorField(mesh, np.full((mesh.n_cells, 2), value, dtype=float))


def _varying_vector_field(mesh, seed):
    rng = np.random.default_rng(seed)
    return VectorField(mesh, rng.uniform(-1.0, 1.0, size=(mesh.n_cells, 2)))


# ---------------------------------------------------------------------------
# ForceContribution
# ---------------------------------------------------------------------------

def test_force_contribution_stores_fields():
    mesh = build_structured_mesh(nx=3, ny=3)
    force = _vector_field(mesh, 1.0)
    contribution = ForceContribution("gravity", force, "N/m^3", {"source": "body force"})
    assert contribution.name == "gravity"
    assert contribution.force is force
    assert contribution.units == "N/m^3"
    assert contribution.metadata == {"source": "body force"}
    assert contribution.mesh is mesh


def test_force_contribution_defaults_metadata_to_empty_dict():
    mesh = build_structured_mesh(nx=3, ny=3)
    contribution = ForceContribution("gravity", _vector_field(mesh, 1.0), "N/m^3")
    assert contribution.metadata == {}


def test_force_contribution_copies_metadata():
    mesh = build_structured_mesh(nx=3, ny=3)
    metadata = {"phase": "liquid"}
    contribution = ForceContribution("gravity", _vector_field(mesh, 1.0), "N/m^3", metadata)
    metadata["phase"] = "gas"
    assert contribution.metadata == {"phase": "liquid"}


def test_force_contribution_magnitude_matches_manual_computation():
    mesh = build_structured_mesh(nx=3, ny=3)
    values = np.column_stack([np.full(mesh.n_cells, 3.0), np.full(mesh.n_cells, 4.0)])
    contribution = ForceContribution("custom", VectorField(mesh, values), "N/m^3")
    assert np.allclose(contribution.magnitude().values, 5.0)


def test_force_contribution_rejects_non_vector_field():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        ForceContribution("gravity", "not a field", "N/m^3")


@pytest.mark.parametrize("bad_name", [123, None, "", "   "])
def test_force_contribution_rejects_invalid_name(bad_name):
    mesh = build_structured_mesh(nx=3, ny=3)
    error = TypeError if not isinstance(bad_name, str) else ValueError
    with pytest.raises(error):
        ForceContribution(bad_name, _vector_field(mesh, 1.0), "N/m^3")


@pytest.mark.parametrize("bad_units", [123, None, "", "   "])
def test_force_contribution_rejects_invalid_units(bad_units):
    mesh = build_structured_mesh(nx=3, ny=3)
    error = TypeError if not isinstance(bad_units, str) else ValueError
    with pytest.raises(error):
        ForceContribution("gravity", _vector_field(mesh, 1.0), bad_units)


def test_force_contribution_rejects_non_dict_metadata():
    mesh = build_structured_mesh(nx=3, ny=3)
    with pytest.raises(TypeError):
        ForceContribution("gravity", _vector_field(mesh, 1.0), "N/m^3", metadata="not a dict")


def test_force_contribution_repr_contains_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    contribution = ForceContribution("gravity", _vector_field(mesh, 1.0), "N/m^3")
    assert "gravity" in repr(contribution)


# ---------------------------------------------------------------------------
# ForceAccounting construction
# ---------------------------------------------------------------------------

def test_force_accounting_rejects_non_mesh():
    with pytest.raises(TypeError):
        ForceAccounting("not a mesh")


def test_force_accounting_starts_empty():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    assert len(accounting) == 0
    assert accounting.names == []


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_stores_and_returns_contribution():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    force = _vector_field(mesh, 1.0)
    contribution = accounting.register("gravity", force, "N/m^3", {"g": 9.81})
    assert isinstance(contribution, ForceContribution)
    assert contribution.name == "gravity"
    assert "gravity" in accounting
    assert accounting.names == ["gravity"]
    assert len(accounting) == 1


def test_register_defaults_units():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    contribution = accounting.register("gravity", _vector_field(mesh, 1.0))
    assert contribution.units == "N/m^3"


def test_register_preserves_insertion_order():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    accounting.register("drag", _vector_field(mesh, 2.0))
    accounting.register("dispersion", _vector_field(mesh, 3.0))
    assert accounting.names == ["gravity", "drag", "dispersion"]


def test_register_rejects_duplicate_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    with pytest.raises(ValueError):
        accounting.register("gravity", _vector_field(mesh, 2.0))


def test_register_rejects_force_on_different_mesh():
    mesh = build_structured_mesh(nx=3, ny=3)
    other_mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(ValueError):
        accounting.register("gravity", _vector_field(other_mesh, 1.0))


def test_register_rejects_wrong_component_count():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    scalar_shaped_force = VectorField(mesh, np.zeros((mesh.n_cells, 1)))
    with pytest.raises(ValueError):
        accounting.register("gravity", scalar_shaped_force)


def test_register_rejects_non_vector_field_force():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(TypeError):
        accounting.register("gravity", "not a field")


def test_register_contribution_rejects_non_contribution():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(TypeError):
        accounting.register_contribution("not a contribution")


def test_register_contribution_accepts_prebuilt_contribution():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    contribution = ForceContribution("gravity", _vector_field(mesh, 1.0), "N/m^3")
    accounting.register_contribution(contribution)
    assert accounting.get("gravity") is contribution


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_get_returns_registered_contribution():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    contribution = accounting.get("gravity")
    assert contribution.name == "gravity"


def test_get_force_returns_vector_field():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    force = _vector_field(mesh, 2.5)
    accounting.register("gravity", force)
    retrieved = accounting.get_force("gravity")
    assert isinstance(retrieved, VectorField)
    assert np.allclose(retrieved.values, 2.5)


def test_get_unknown_name_raises_key_error():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(KeyError):
        accounting.get("does_not_exist")


def test_get_rejects_non_string_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(TypeError):
        accounting.get(123)


# ---------------------------------------------------------------------------
# Total force
# ---------------------------------------------------------------------------

def test_total_force_sums_registered_contributions():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    accounting.register("drag", _vector_field(mesh, 2.0))
    total = accounting.total_force()
    assert isinstance(total, VectorField)
    assert np.allclose(total.values, 3.0)


def test_total_force_with_single_contribution_equals_that_force():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    force = _varying_vector_field(mesh, seed=1)
    accounting.register("gravity", force)
    total = accounting.total_force()
    assert np.allclose(total.values, force.values)


def test_total_force_with_no_contributions_raises():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(ValueError):
        accounting.total_force()


def test_total_force_can_cancel_out():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    accounting.register("buoyancy", _vector_field(mesh, -1.0))
    total = accounting.total_force()
    assert np.allclose(total.values, 0.0)


# ---------------------------------------------------------------------------
# Magnitudes
# ---------------------------------------------------------------------------

def test_magnitude_matches_manual_computation():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    values = np.column_stack([np.full(mesh.n_cells, 3.0), np.full(mesh.n_cells, 4.0)])
    accounting.register("gravity", VectorField(mesh, values))
    assert np.allclose(accounting.magnitude("gravity").values, 5.0)


def test_magnitudes_returns_every_contribution():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    accounting.register("drag", _vector_field(mesh, 2.0))
    magnitudes = accounting.magnitudes()
    assert set(magnitudes.keys()) == {"gravity", "drag"}
    assert np.allclose(magnitudes["gravity"].values, np.sqrt(2.0))
    assert np.allclose(magnitudes["drag"].values, np.sqrt(8.0))


def test_total_magnitude_matches_total_force_magnitude():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 3.0))
    accounting.register("drag", _vector_field(mesh, -1.0))
    expected = np.sqrt(np.sum(accounting.total_force().values ** 2, axis=1))
    assert np.allclose(accounting.total_magnitude().values, expected)


# ---------------------------------------------------------------------------
# Normalized contributions
# ---------------------------------------------------------------------------

def test_normalized_contributions_sum_to_one():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _varying_vector_field(mesh, seed=2))
    accounting.register("drag", _varying_vector_field(mesh, seed=3))
    accounting.register("dispersion", _varying_vector_field(mesh, seed=4))
    normalized = accounting.normalized_contributions()
    total_share = sum(field.values for field in normalized.values())
    assert np.allclose(total_share, 1.0)


def test_normalized_contributions_equal_split_for_equal_forces():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    accounting.register("drag", _vector_field(mesh, 1.0))
    normalized = accounting.normalized_contributions()
    assert np.allclose(normalized["gravity"].values, 0.5)
    assert np.allclose(normalized["drag"].values, 0.5)


def test_normalized_contributions_single_force_is_fully_dominant():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 5.0))
    normalized = accounting.normalized_contributions()
    assert np.allclose(normalized["gravity"].values, 1.0)


def test_normalized_contributions_stays_finite_when_forces_are_zero():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 0.0))
    accounting.register("drag", _vector_field(mesh, 0.0))
    normalized = accounting.normalized_contributions()
    assert np.all(np.isfinite(normalized["gravity"].values))
    assert np.all(np.isfinite(normalized["drag"].values))


def test_normalized_contributions_with_no_contributions_raises():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(ValueError):
        accounting.normalized_contributions()


# ---------------------------------------------------------------------------
# Summary / report
# ---------------------------------------------------------------------------

def test_summary_contains_expected_statistics():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0), "N/m^3", {"g": 9.81})
    summary = accounting.summary()
    stats = summary["gravity"]
    assert stats["units"] == "N/m^3"
    assert np.isclose(stats["mean_magnitude"], np.sqrt(2.0))
    assert np.isclose(stats["max_magnitude"], np.sqrt(2.0))
    assert np.isclose(stats["min_magnitude"], np.sqrt(2.0))
    assert np.isclose(stats["mean_normalized_contribution"], 1.0)
    assert stats["metadata"] == {"g": 9.81}


def test_summary_with_no_contributions_raises():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    with pytest.raises(ValueError):
        accounting.summary()


def test_report_lists_every_contribution_name():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    accounting.register("drag", _vector_field(mesh, 2.0))
    text = accounting.report()
    assert isinstance(text, str)
    assert "gravity" in text
    assert "drag" in text


def test_report_with_no_contributions_is_still_a_string():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    text = accounting.report()
    assert isinstance(text, str)
    assert "no force contributions" in text


def test_repr_contains_contribution_count_and_names():
    mesh = build_structured_mesh(nx=3, ny=3)
    accounting = ForceAccounting(mesh)
    accounting.register("gravity", _vector_field(mesh, 1.0))
    text = repr(accounting)
    assert "1" in text
    assert "gravity" in text


# ---------------------------------------------------------------------------
# Arbitrary force contributions (gravity, drag, packed-bed, capillary,
# dispersion, user-defined), built with existing framework infrastructure.
# ---------------------------------------------------------------------------

def test_supports_full_range_of_force_contributions():
    mesh = build_structured_mesh(nx=4, ny=4)
    accounting = ForceAccounting(mesh)

    # Gravity: a uniform body force per unit volume, rho * g.
    gas = Phase("gas", mesh, density=1.2, viscosity=1.8e-5, volume_fraction=0.4)
    liquid = Phase("liquid", mesh, density=1000.0, viscosity=1e-3, volume_fraction=0.6)
    gravity_force = VectorField(mesh, np.column_stack([
        np.zeros(mesh.n_cells),
        np.full(mesh.n_cells, -liquid.density * 9.81),
    ]))
    accounting.register("gravity", gravity_force, "N/m^3", {"phase": "liquid"})

    # Interphase drag: reuse the Schiller-Naumann drag coefficient directly.
    gas_velocity = _vector_field(mesh, 0.5)
    liquid_velocity = _vector_field(mesh, 0.1)
    relative_speed = relative_velocity_magnitude(gas_velocity, liquid_velocity)
    drag_coefficient = interphase_drag_coefficient(
        mesh, liquid.density, liquid.viscosity, gas.volume_fraction, relative_speed, particle_diameter=1e-3
    )
    drag_force = VectorField(
        mesh, drag_coefficient.values[:, None] * (liquid_velocity.values - gas_velocity.values)
    )
    accounting.register("interphase_drag", drag_force, "N/m^3", {"model": "schiller_naumann"})

    # Packed-bed resistance: reuse the Ergun resistance model directly.
    bed_properties = PackedBedProperties(
        particle_diameter=1e-3, porosity=0.4, fluid_viscosity=liquid.viscosity, fluid_density=liquid.density
    )
    resistance_model = ErgunResistanceModel(mesh, bed_properties)
    packed_bed_force = resistance_model.momentum_source(liquid_velocity)
    accounting.register("packed_bed_resistance", packed_bed_force, "N/m^3", {"model": "ergun"})

    # Capillary pressure: turn the scalar closure into a force via its gradient.
    capillary_model = ConstantCapillaryPressureModel(mesh, capillary_pressure_value=500.0)
    capillary_pressure_field = capillary_pressure_from_phase(
        capillary_model, liquid, bed_properties, surface_tension=0.072, contact_angle=0.3
    )
    capillary_force = gradient(capillary_pressure_field)
    accounting.register("capillary_pressure", capillary_force, "N/m^3", {"model": "constant"})

    # Mechanical dispersion: an illustrative velocity-proportional force.
    dispersion_coefficient = 0.05
    dispersion_force = VectorField(mesh, -dispersion_coefficient * liquid_velocity.values)
    accounting.register("mechanical_dispersion", dispersion_force, "N/m^3", {"coefficient": dispersion_coefficient})

    # User-defined: an arbitrary, package-agnostic force.
    user_force = VectorField(mesh, np.column_stack([
        np.full(mesh.n_cells, 2.0), np.full(mesh.n_cells, -3.0),
    ]))
    accounting.register("custom_electromagnetic_force", user_force, "N/m^3", {"author": "user"})

    expected_names = {
        "gravity",
        "interphase_drag",
        "packed_bed_resistance",
        "capillary_pressure",
        "mechanical_dispersion",
        "custom_electromagnetic_force",
    }
    assert set(accounting.names) == expected_names
    assert len(accounting) == 6

    total = accounting.total_force()
    expected_total = sum(
        (accounting.get_force(name).values for name in expected_names), np.zeros((mesh.n_cells, 2))
    )
    assert np.allclose(total.values, expected_total)

    normalized = accounting.normalized_contributions()
    total_share = sum(field.values for field in normalized.values())
    assert np.allclose(total_share, 1.0)

    report_text = accounting.report()
    for name in expected_names:
        assert name in report_text
