import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ScalarField, VectorField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.equation import Equation
from src.cfd.pressure_correction import solve_pressure_correction
from src.cfd.simple_solver import (
    SimpleSolverResult,
    assemble_momentum_system,
    compute_residuals,
    correct_velocity,
    has_converged,
    momentum_relaxation_scale,
    simple_continuity_equation,
    simple_momentum_equation,
    solve_momentum_predictor,
    solve_pressure_correction_step,
    solve_simple,
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


# ---------------------------------------------------------------------------
# Basic solving
# ---------------------------------------------------------------------------

def test_zero_boundary_conditions_give_zero_velocity_and_pressure():
    mesh = build_structured_mesh(nx=5, ny=5)
    result = solve_simple(
        mesh, _u_boundaries(), _v_boundaries(), viscosity=1.0,
        max_outer_iterations=10, outer_tolerance=1e-8,
    )
    assert result.converged is True
    assert np.allclose(result.velocity.values, 0.0, atol=1e-8)
    assert np.allclose(result.pressure.values, 0.0, atol=1e-8)


def test_uniform_boundary_velocity_gives_uniform_field():
    # A velocity that is the same on every boundary, with no obstruction, has
    # zero divergence and needs no pressure gradient to sustain it, so the
    # steady solution is that same uniform velocity everywhere and p = 0.
    mesh = build_structured_mesh(nx=6, ny=6)
    result = solve_simple(
        mesh,
        _u_boundaries(left=1.0, right=1.0, top=1.0, bottom=1.0),
        _v_boundaries(),
        viscosity=1.0,
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
    result = solve_simple(
        mesh,
        _u_boundaries(top=1.0),
        _v_boundaries(),
        viscosity=0.1,
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
    result = solve_simple(
        mesh, _u_boundaries(top=1.0), _v_boundaries(), viscosity=0.5, max_outer_iterations=8,
    )
    assert len(result.residual_history) == result.iterations_run
    assert result.iterations_run <= 8


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def test_solve_simple_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        solve_simple("not a mesh", _u_boundaries(), _v_boundaries(), 1.0)


def test_solve_simple_rejects_3d_mesh():
    mesh_3d = Mesh(
        cell_centers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_centers=np.array([[0.5, 0.0, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    with pytest.raises(ValueError):
        solve_simple(mesh_3d, _u_boundaries(), _v_boundaries(), 1.0)


def test_solve_simple_rejects_non_uniform_spacing():
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
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0)


# ---------------------------------------------------------------------------
# Boundary condition validation
# ---------------------------------------------------------------------------

def test_missing_u_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    incomplete = [FixedValueBC("left", 0.0), FixedValueBC("right", 0.0), FixedValueBC("top", 0.0)]
    with pytest.raises(ValueError):
        solve_simple(mesh, incomplete, _v_boundaries(), 1.0)


def test_duplicate_v_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    duplicated = [
        FixedValueBC("left", 0.0), FixedValueBC("left", 1.0),
        FixedValueBC("top", 0.0), FixedValueBC("bottom", 0.0),
    ]
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), duplicated, 1.0)


def test_non_fixed_value_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = [
        FixedValueBC("left", 0.0), FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0), ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        solve_simple(mesh, boundaries, _v_boundaries(), 1.0)


def test_wrong_number_of_boundaries_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries()[:3], _v_boundaries(), 1.0)


def test_boundary_conditions_must_be_list_or_tuple():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_simple(mesh, FixedValueBC("left", 0.0), _v_boundaries(), 1.0)


# ---------------------------------------------------------------------------
# viscosity validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_viscosity", [True, "1.0", None])
def test_non_numeric_viscosity_raises_type_error(bad_viscosity):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), bad_viscosity)


def test_non_positive_viscosity_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 0.0)


# ---------------------------------------------------------------------------
# Relaxation factor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0.0, 1.5, -0.2])
def test_invalid_velocity_relaxation_raises_value_error(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, velocity_relaxation=bad_value)


@pytest.mark.parametrize("bad_value", [0.0, 1.5, -0.2])
def test_invalid_pressure_relaxation_raises_value_error(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, pressure_relaxation=bad_value)


@pytest.mark.parametrize("bad_value", [True, "0.5", None])
def test_non_numeric_relaxation_raises_type_error(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, velocity_relaxation=bad_value)


# ---------------------------------------------------------------------------
# max_outer_iterations / outer_tolerance / linear_method validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [True, 1.5, "10", None])
def test_non_integer_max_outer_iterations_raises_type_error(bad_value):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, max_outer_iterations=bad_value)


def test_non_positive_max_outer_iterations_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, max_outer_iterations=0)


def test_non_positive_outer_tolerance_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, outer_tolerance=0.0)


def test_unknown_linear_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_simple(
            mesh, _u_boundaries(), _v_boundaries(), 1.0, linear_method="fancy", max_outer_iterations=1,
        )


# ---------------------------------------------------------------------------
# initial_velocity / initial_pressure validation
# ---------------------------------------------------------------------------

def test_initial_velocity_wrong_type_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, initial_velocity="not a field")


def test_initial_velocity_wrong_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    bad_velocity = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, initial_velocity=bad_velocity)


def test_initial_velocity_wrong_component_count_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    bad_velocity = VectorField(mesh, np.zeros((mesh.n_cells, 3)))
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, initial_velocity=bad_velocity)


def test_initial_pressure_wrong_type_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, initial_pressure="not a field")


def test_initial_pressure_wrong_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    bad_pressure = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        solve_simple(mesh, _u_boundaries(), _v_boundaries(), 1.0, initial_pressure=bad_pressure)


# ---------------------------------------------------------------------------
# Momentum prediction (assemble_momentum_system / solve_momentum_predictor)
# ---------------------------------------------------------------------------

def test_invalid_component_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        assemble_momentum_system(mesh, velocity, _u_boundaries(), 1.0, pressure, "w")


def test_pressure_must_be_scalar_field():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    with pytest.raises(TypeError):
        assemble_momentum_system(mesh, velocity, _u_boundaries(), 1.0, "not a field", "u")


def test_pressure_wrong_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    pressure = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        assemble_momentum_system(mesh, velocity, _u_boundaries(), 1.0, pressure, "u")


def test_solve_momentum_predictor_returns_expected_shapes():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))

    velocity_star, u_system, v_system = solve_momentum_predictor(
        mesh, velocity, _u_boundaries(top=1.0), _v_boundaries(), 1.0, pressure,
    )

    assert velocity_star.values.shape == (mesh.n_cells, 2)
    assert u_system.size == mesh.n_cells
    assert v_system.size == mesh.n_cells


# ---------------------------------------------------------------------------
# momentum_relaxation_scale
# ---------------------------------------------------------------------------

def test_momentum_relaxation_scale_returns_positive_float():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    _, u_system, v_system = solve_momentum_predictor(
        mesh, velocity, _u_boundaries(), _v_boundaries(), 1.0, pressure,
    )
    scale = momentum_relaxation_scale(mesh, u_system, v_system, _u_boundaries(), _v_boundaries())
    assert scale > 0.0


def test_momentum_relaxation_scale_rejects_non_linear_system():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        momentum_relaxation_scale(mesh, "not a system", "not a system", _u_boundaries(), _v_boundaries())


# ---------------------------------------------------------------------------
# Pressure correction step
# ---------------------------------------------------------------------------

def test_solve_pressure_correction_step_matches_direct_call():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())

    direct = solve_pressure_correction(
        mesh,
        velocity_star,
        [ZeroGradientBC("left"), ZeroGradientBC("right"), ZeroGradientBC("top"), ZeroGradientBC("bottom")],
        dt=2.0,
        tolerance=1e-10,
        max_iterations=20000,
    )
    wrapped = solve_pressure_correction_step(
        mesh, velocity_star, relaxation_scale=2.0, tolerance=1e-10, max_iterations=20000,
    )

    assert np.allclose(direct.values, wrapped.values, atol=1e-8)


def test_solve_pressure_correction_step_rejects_non_positive_scale():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        solve_pressure_correction_step(mesh, velocity_star, relaxation_scale=0.0)


# ---------------------------------------------------------------------------
# Velocity correction
# ---------------------------------------------------------------------------

def test_correct_velocity_reapplies_boundary_values():
    mesh = build_structured_mesh(nx=6, ny=6)
    u_boundaries = _u_boundaries(top=1.0)
    v_boundaries = _v_boundaries()
    velocity_star = VectorField(mesh, np.tile([0.5, 0.5], (mesh.n_cells, 1)))
    p_prime = ScalarField(mesh, np.zeros(mesh.n_cells))

    corrected = correct_velocity(mesh, velocity_star, p_prime, 1.0, u_boundaries, v_boundaries)

    coords = mesh.cell_centers
    top_indices = np.where(np.isclose(coords[:, 1], coords[:, 1].max()))[0]
    bottom_indices = np.where(np.isclose(coords[:, 1], coords[:, 1].min()))[0]

    assert np.allclose(corrected.values[top_indices, 0], 1.0)
    assert np.allclose(corrected.values[bottom_indices, 0], 0.0)
    assert np.allclose(corrected.values[top_indices, 1], 0.0)


def test_correct_velocity_rejects_non_positive_scale():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    p_prime = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        correct_velocity(mesh, velocity_star, p_prime, 0.0, _u_boundaries(), _v_boundaries())


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def test_compute_residuals_returns_expected_keys():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    residuals = compute_residuals(mesh, velocity, velocity)
    assert set(residuals.keys()) == {"continuity", "u_momentum", "v_momentum"}


def test_compute_residuals_zero_when_velocity_unchanged():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    residuals = compute_residuals(mesh, velocity, velocity)
    assert residuals["u_momentum"] == 0.0
    assert residuals["v_momentum"] == 0.0
    assert residuals["continuity"] == 0.0


def test_compute_residuals_rejects_wrong_mesh():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    other_velocity = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        compute_residuals(mesh, velocity, other_velocity)


# ---------------------------------------------------------------------------
# Convergence checking
# ---------------------------------------------------------------------------

def test_has_converged_true_when_all_below_tolerance():
    residuals = {"continuity": 1e-9, "u_momentum": 1e-9, "v_momentum": 1e-9}
    assert has_converged(residuals, 1e-6) is True


def test_has_converged_false_when_any_above_tolerance():
    residuals = {"continuity": 1e-9, "u_momentum": 1e-3, "v_momentum": 1e-9}
    assert has_converged(residuals, 1e-6) is False


def test_has_converged_rejects_non_dict():
    with pytest.raises(TypeError):
        has_converged(["not", "a", "dict"], 1e-6)


def test_has_converged_rejects_non_positive_tolerance():
    with pytest.raises(ValueError):
        has_converged({"continuity": 0.0}, 0.0)


# ---------------------------------------------------------------------------
# SimpleSolverResult
# ---------------------------------------------------------------------------

def test_simple_solver_result_repr_contains_key_fields():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    pressure = ScalarField(mesh, np.zeros(mesh.n_cells))
    result = SimpleSolverResult(velocity, pressure, 3, True, [{"continuity": 0.0}])
    text = repr(result)
    assert "iterations_run=3" in text
    assert "converged=True" in text


# ---------------------------------------------------------------------------
# Equation description
# ---------------------------------------------------------------------------

def test_simple_momentum_equation_returns_equation_instance():
    equation = simple_momentum_equation()
    assert isinstance(equation, Equation)
    assert equation.lhs == "convection(U)"
    assert equation.rhs == "viscosity * laplacian(U) - gradient(p)"
    assert str(equation) == "convection(U) = viscosity * laplacian(U) - gradient(p)"


def test_simple_continuity_equation_returns_equation_instance():
    equation = simple_continuity_equation()
    assert isinstance(equation, Equation)
    assert equation.lhs == "divergence(U)"
    assert equation.rhs == "0"
    assert str(equation) == "divergence(U) = 0"
