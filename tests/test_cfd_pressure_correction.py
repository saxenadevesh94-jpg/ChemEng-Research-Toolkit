import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, VectorField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.equation import Equation
from src.cfd.pressure_correction import (
    assemble_pressure_correction,
    pressure_correction_equation,
    solve_pressure_correction,
)
from src.cfd.solvers import GaussSeidelSolver, JacobiSolver


def _neumann_boundaries():
    return [
        ZeroGradientBC("left"),
        ZeroGradientBC("right"),
        ZeroGradientBC("top"),
        ZeroGradientBC("bottom"),
    ]


def _constant_velocity_field(mesh, u=2.0, v=-1.0):
    return VectorField(mesh, np.tile([u, v], (mesh.n_cells, 1)))


# ---------------------------------------------------------------------------
# Basic solving
# ---------------------------------------------------------------------------

def test_square_domain_solves_without_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    field = solve_pressure_correction(mesh, velocity_star, _neumann_boundaries())
    assert field.values.shape[0] == mesh.n_cells


def test_square_domain_matches_mesh():
    mesh = build_structured_mesh(nx=6, ny=6)
    velocity_star = _constant_velocity_field(mesh)
    field = solve_pressure_correction(mesh, velocity_star, _neumann_boundaries())
    assert field.mesh is mesh


def test_divergence_free_velocity_gives_zero_pressure_correction():
    # A spatially constant velocity field has zero divergence everywhere, so
    # the pressure correction source term is zero and, combined with
    # homogeneous Neumann boundaries and a single pinned reference cell, the
    # only solution is p_prime = 0 everywhere.
    mesh = build_structured_mesh(nx=6, ny=6)
    velocity_star = _constant_velocity_field(mesh, u=3.0, v=1.5)
    field = solve_pressure_correction(
        mesh, velocity_star, _neumann_boundaries(), tolerance=1e-10, max_iterations=20000,
    )
    assert np.allclose(field.values, 0.0, atol=1e-6)


def test_reference_cell_is_pinned_to_zero():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())
    field = solve_pressure_correction(mesh, velocity_star, _neumann_boundaries(), reference_cell=7)
    assert np.isclose(field.values[7], 0.0, atol=1e-8)


def test_solution_satisfies_assembled_system():
    mesh = build_structured_mesh(nx=6, ny=6)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())
    system = assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries())
    solver = GaussSeidelSolver(max_iterations=20000, tolerance=1e-10)
    solution = solver.solve(system)

    residual = system.matrix.to_array() @ solution - system.rhs
    assert np.allclose(residual, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# dt and reference_cell parameters
# ---------------------------------------------------------------------------

def test_smaller_dt_scales_pressure_correction_proportionally():
    mesh = build_structured_mesh(nx=6, ny=6)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())

    field_dt_1 = solve_pressure_correction(
        mesh, velocity_star, _neumann_boundaries(), dt=1.0, tolerance=1e-10, max_iterations=20000,
    )
    field_dt_half = solve_pressure_correction(
        mesh, velocity_star, _neumann_boundaries(), dt=0.5, tolerance=1e-10, max_iterations=20000,
    )

    # Halving dt doubles the source term, and the assembled system is linear
    # in the source term, so the solution should double too.
    assert np.allclose(field_dt_half.values, 2.0 * field_dt_1.values, atol=1e-6)


@pytest.mark.parametrize("bad_dt", [True, "1.0", None])
def test_non_numeric_dt_raises_type_error(bad_dt):
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(TypeError):
        assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries(), dt=bad_dt)


def test_non_positive_dt_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries(), dt=0.0)


@pytest.mark.parametrize("bad_reference_cell", [True, 1.0, "0", None])
def test_non_integer_reference_cell_raises_type_error(bad_reference_cell):
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(TypeError):
        assemble_pressure_correction(
            mesh, velocity_star, _neumann_boundaries(), reference_cell=bad_reference_cell
        )


@pytest.mark.parametrize("bad_reference_cell", [-1, 25, 1000])
def test_out_of_range_reference_cell_raises_value_error(bad_reference_cell):
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(ValueError):
        assemble_pressure_correction(
            mesh, velocity_star, _neumann_boundaries(), reference_cell=bad_reference_cell
        )


# ---------------------------------------------------------------------------
# velocity_star validation
# ---------------------------------------------------------------------------

def test_velocity_star_must_be_vector_field():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_pressure_correction(mesh, (1.0, 0.0), _neumann_boundaries())


def test_velocity_star_on_different_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries())


def test_velocity_star_with_wrong_component_count_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, np.zeros((mesh.n_cells, 3)))
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries())


# ---------------------------------------------------------------------------
# Solver methods
# ---------------------------------------------------------------------------

def test_jacobi_and_gauss_seidel_agree_on_solution():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())

    field_jacobi = solve_pressure_correction(
        mesh, velocity_star, _neumann_boundaries(), method="jacobi", tolerance=1e-10, max_iterations=20000,
    )
    field_gauss_seidel = solve_pressure_correction(
        mesh, velocity_star, _neumann_boundaries(), method="gauss_seidel",
        tolerance=1e-10, max_iterations=20000,
    )

    assert np.allclose(field_jacobi.values, field_gauss_seidel.values, atol=1e-4)


def test_default_method_is_gauss_seidel():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())
    system = assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries())

    default_field = solve_pressure_correction(
        mesh, velocity_star, _neumann_boundaries(), tolerance=1e-10, max_iterations=20000,
    )
    gs_solution = GaussSeidelSolver(max_iterations=20000, tolerance=1e-10).solve(system)

    assert np.allclose(default_field.values, gs_solution, atol=1e-8)


def test_unknown_solver_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(ValueError):
        solve_pressure_correction(mesh, velocity_star, _neumann_boundaries(), method="conjugate_gradient")


def test_solvers_converge_within_tolerance():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = VectorField(mesh, mesh.cell_centers.copy())
    system = assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries())

    solver = JacobiSolver(max_iterations=5000, tolerance=1e-8)
    solver.solve(system)

    assert solver.converged is True
    assert solver.residual_history[-1] < 1e-8


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def test_assemble_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        assemble_pressure_correction("not a mesh", (1.0, 0.0), _neumann_boundaries())


def test_assemble_rejects_3d_mesh():
    mesh_3d = Mesh(
        cell_centers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_centers=np.array([[0.5, 0.0, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    velocity_star = VectorField(mesh_3d, np.zeros((mesh_3d.n_cells, 2)))
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh_3d, velocity_star, _neumann_boundaries())


def test_assemble_rejects_non_uniform_spacing():
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
    velocity_star = VectorField(mesh, np.zeros((mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries())


# ---------------------------------------------------------------------------
# Boundary condition validation
# ---------------------------------------------------------------------------

def test_missing_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    incomplete = [
        ZeroGradientBC("left"),
        ZeroGradientBC("right"),
        ZeroGradientBC("top"),
    ]
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, incomplete)


def test_duplicate_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    duplicated = [
        ZeroGradientBC("left"),
        ZeroGradientBC("left"),
        ZeroGradientBC("top"),
        ZeroGradientBC("bottom"),
    ]
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, duplicated)


def test_non_zero_gradient_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    boundaries = [
        ZeroGradientBC("left"),
        ZeroGradientBC("right"),
        ZeroGradientBC("top"),
        FixedValueBC("bottom", 0.0),
    ]
    with pytest.raises(TypeError):
        assemble_pressure_correction(mesh, velocity_star, boundaries)


def test_wrong_number_of_boundaries_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(ValueError):
        assemble_pressure_correction(mesh, velocity_star, _neumann_boundaries()[:3])


def test_boundary_conditions_must_be_list_or_tuple():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_star = _constant_velocity_field(mesh)
    with pytest.raises(TypeError):
        assemble_pressure_correction(mesh, velocity_star, ZeroGradientBC("left"))


# ---------------------------------------------------------------------------
# Equation description
# ---------------------------------------------------------------------------

def test_pressure_correction_equation_returns_equation_instance():
    equation = pressure_correction_equation()
    assert isinstance(equation, Equation)
    assert equation.lhs == "laplacian(p_prime)"
    assert equation.rhs == "divergence(velocity_star) / dt"
    assert str(equation) == "laplacian(p_prime) = divergence(velocity_star) / dt"
