import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ScalarField, VectorField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.navier_stokes import NavierStokesResult, solve_navier_stokes


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


# ---------------------------------------------------------------------------
# Basic solving
# ---------------------------------------------------------------------------

def test_zero_boundary_conditions_give_zero_velocity_and_reference_pressure():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = solve_navier_stokes(
        mesh,
        _u_boundaries(),
        _v_boundaries(),
        density=1.0,
        dynamic_viscosity=1.0,
        pressure_reference_value=2.0,
        max_outer_iterations=10,
        outer_tolerance=1e-8,
    )
    assert result.converged is True
    assert np.allclose(result.velocity.values, 0.0, atol=1e-8)
    assert np.allclose(result.pressure.values, 2.0, atol=1e-6)


def test_uniform_boundary_velocity_gives_uniform_field():
    # A velocity that is the same on every boundary, with no obstruction, has
    # zero divergence and needs no pressure gradient to sustain it, so the
    # steady solution is that same uniform velocity everywhere and a uniform
    # pressure field equal to the requested reference value.
    mesh = build_structured_mesh(nx=6, ny=6)
    result = solve_navier_stokes(
        mesh,
        _u_boundaries(left=1.0, right=1.0, top=1.0, bottom=1.0),
        _v_boundaries(),
        density=1.2,
        dynamic_viscosity=1.2,
        velocity_relaxation=1.0,
        pressure_relaxation=1.0,
        max_outer_iterations=20,
        outer_tolerance=1e-6,
    )
    assert np.allclose(result.velocity.values[:, 0], 1.0, atol=1e-4)
    assert np.allclose(result.velocity.values[:, 1], 0.0, atol=1e-4)
    assert np.allclose(result.pressure.values, 0.0, atol=1e-4)


def test_lid_driven_cavity_runs_without_error():
    mesh = build_structured_mesh(nx=6, ny=6)
    result = solve_navier_stokes(
        mesh,
        _u_boundaries(top=1.0),
        _v_boundaries(),
        density=1.0,
        dynamic_viscosity=0.1,
        max_outer_iterations=15,
    )
    assert result.velocity.values.shape == (mesh.n_cells, 2)
    assert result.pressure.values.shape == (mesh.n_cells,)
    assert np.all(np.isfinite(result.velocity.values))
    assert np.all(np.isfinite(result.pressure.values))
    assert result.velocity.mesh is mesh
    assert result.pressure.mesh is mesh


def test_residual_history_length_matches_iterations_run():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = solve_navier_stokes(
        mesh,
        _u_boundaries(top=1.0),
        _v_boundaries(),
        density=1.0,
        dynamic_viscosity=0.5,
        max_outer_iterations=8,
    )
    assert len(result.residual_history) == result.iterations_run
    assert result.iterations_run <= 8


def test_result_stores_density_and_dynamic_viscosity():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = solve_navier_stokes(
        mesh, _u_boundaries(), _v_boundaries(), density=2.0, dynamic_viscosity=0.4, max_outer_iterations=5,
    )
    assert result.density == 2.0
    assert result.dynamic_viscosity == 0.4


# ---------------------------------------------------------------------------
# Pressure reference fixing
# ---------------------------------------------------------------------------

def test_pressure_reference_cell_matches_requested_value():
    mesh = build_structured_mesh(nx=6, ny=6)
    reference_cell = 7
    reference_value = 5.0
    result = solve_navier_stokes(
        mesh,
        _u_boundaries(top=1.0),
        _v_boundaries(),
        density=1.0,
        dynamic_viscosity=0.5,
        pressure_reference_cell=reference_cell,
        pressure_reference_value=reference_value,
        max_outer_iterations=15,
    )
    assert np.isclose(result.pressure.values[reference_cell], reference_value, atol=1e-8)


def test_different_reference_cells_give_same_pressure_gradient():
    mesh = build_structured_mesh(nx=6, ny=6)
    result_a = solve_navier_stokes(
        mesh,
        _u_boundaries(top=1.0),
        _v_boundaries(),
        density=1.0,
        dynamic_viscosity=0.5,
        pressure_reference_cell=0,
        pressure_reference_value=0.0,
        max_outer_iterations=15,
    )
    result_b = solve_navier_stokes(
        mesh,
        _u_boundaries(top=1.0),
        _v_boundaries(),
        density=1.0,
        dynamic_viscosity=0.5,
        pressure_reference_cell=10,
        pressure_reference_value=3.0,
        max_outer_iterations=15,
    )
    difference = result_a.pressure.values - result_b.pressure.values
    assert np.allclose(difference, difference[0], atol=1e-6)


def test_invalid_pressure_reference_cell_type_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, pressure_reference_cell="0",
        )


def test_pressure_reference_cell_out_of_range_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, pressure_reference_cell=mesh.n_cells,
        )


def test_negative_pressure_reference_cell_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, pressure_reference_cell=-1,
        )


@pytest.mark.parametrize("bad_value", [True, "0.0", None])
def test_non_numeric_pressure_reference_value_raises_type_error(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, pressure_reference_value=bad_value,
        )


# ---------------------------------------------------------------------------
# Mesh validation (delegated to solve_simple, still exercised here)
# ---------------------------------------------------------------------------

def test_solve_navier_stokes_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        solve_navier_stokes("not a mesh", _u_boundaries(), _v_boundaries(), 1.0, 1.0)


def test_solve_navier_stokes_rejects_3d_mesh():
    mesh_3d = Mesh(
        cell_centers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_centers=np.array([[0.5, 0.0, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    with pytest.raises(ValueError):
        solve_navier_stokes(mesh_3d, _u_boundaries(), _v_boundaries(), 1.0, 1.0)


def test_solve_navier_stokes_rejects_non_uniform_spacing():
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
    with pytest.raises(ValueError):
        solve_navier_stokes(mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0)


# ---------------------------------------------------------------------------
# Boundary condition validation (delegated to solve_simple)
# ---------------------------------------------------------------------------

def test_missing_u_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    incomplete = [FixedValueBC("left", 0.0), FixedValueBC("right", 0.0), FixedValueBC("top", 0.0)]
    with pytest.raises(ValueError):
        solve_navier_stokes(mesh, incomplete, _v_boundaries(), 1.0, 1.0)


def test_non_fixed_value_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = [
        FixedValueBC("left", 0.0), FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0), ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        solve_navier_stokes(mesh, boundaries, _v_boundaries(), 1.0, 1.0)


# ---------------------------------------------------------------------------
# density / dynamic_viscosity validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_density", [True, "1.0", None])
def test_non_numeric_density_raises_type_error(bad_density):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_navier_stokes(mesh, _u_boundaries(), _v_boundaries(), bad_density, 1.0)


def test_non_positive_density_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(mesh, _u_boundaries(), _v_boundaries(), 0.0, 1.0)


@pytest.mark.parametrize("bad_viscosity", [True, "1.0", None])
def test_non_numeric_dynamic_viscosity_raises_type_error(bad_viscosity):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_navier_stokes(mesh, _u_boundaries(), _v_boundaries(), 1.0, bad_viscosity)


def test_non_positive_dynamic_viscosity_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(mesh, _u_boundaries(), _v_boundaries(), 1.0, 0.0)


# ---------------------------------------------------------------------------
# initial_velocity / initial_pressure validation
# ---------------------------------------------------------------------------

def test_initial_velocity_wrong_type_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, initial_velocity="not a field",
        )


def test_initial_pressure_wrong_type_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, initial_pressure="not a field",
        )


def test_initial_pressure_wrong_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    bad_pressure = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, initial_pressure=bad_pressure,
        )


def test_initial_pressure_is_converted_to_kinematic_units_before_solving():
    # A uniform, non-zero initial pressure guess should not change the
    # steady zero-velocity solution for zero boundary velocities, and the
    # final field should still land on the requested reference value.
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_pressure = ScalarField(mesh, np.full(mesh.n_cells, 10.0))
    result = solve_navier_stokes(
        mesh,
        _u_boundaries(),
        _v_boundaries(),
        density=2.0,
        dynamic_viscosity=1.0,
        initial_pressure=initial_pressure,
        pressure_reference_value=0.0,
        max_outer_iterations=10,
    )
    assert np.allclose(result.velocity.values, 0.0, atol=1e-8)
    assert np.allclose(result.pressure.values, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Relaxation / iteration / method validation (delegated to solve_simple)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0.0, 1.5, -0.2])
def test_invalid_velocity_relaxation_raises_value_error(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, velocity_relaxation=bad_value,
        )


def test_non_positive_max_outer_iterations_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, max_outer_iterations=0,
        )


def test_unknown_linear_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_navier_stokes(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, 1.0, linear_method="fancy", max_outer_iterations=1,
        )


# ---------------------------------------------------------------------------
# NavierStokesResult
# ---------------------------------------------------------------------------

def test_navier_stokes_result_repr_contains_key_fields():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    result = NavierStokesResult(velocity, pressure, 1.0, 1.0, 3, True, [{"continuity": 0.0}])
    text = repr(result)
    assert "iterations_run=3" in text
    assert "converged=True" in text
