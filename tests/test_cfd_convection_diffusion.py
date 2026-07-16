import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ScalarField, VectorField, ZeroGradientBC
from src.cfd.convection_diffusion import (
    assemble_convection_diffusion_system,
    convection_diffusion_equation,
    solve_convection_diffusion,
)
from src.cfd.diffusion_solver import build_structured_mesh, solve_diffusion
from src.cfd.equation import Equation
from src.cfd.solvers import GaussSeidelSolver, JacobiSolver


def _square_boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


# ---------------------------------------------------------------------------
# Basic solving
# ---------------------------------------------------------------------------

def test_square_domain_solves_without_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_convection_diffusion(
        mesh, (1.0, 0.0), _square_boundaries(left=1.0, right=1.0, top=1.0, bottom=1.0), diffusivity=1.0,
    )
    assert field.values.shape[0] == mesh.n_cells


def test_square_domain_matches_mesh():
    mesh = build_structured_mesh(nx=6, ny=6)
    field = solve_convection_diffusion(mesh, (1.0, 1.0), _square_boundaries(), diffusivity=1.0)
    assert field.mesh is mesh


def test_zero_velocity_matches_pure_diffusion_solution():
    # With u = v = 0 everywhere, convection-diffusion reduces exactly to the
    # plain diffusion equation already solved in Sprint 18.
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=1.0, right=5.0, top=3.0, bottom=7.0)

    convection_field = solve_convection_diffusion(
        mesh, (0.0, 0.0), boundaries, diffusivity=1.0, tolerance=1e-10, max_iterations=20000,
    )
    diffusion_field = solve_diffusion(mesh, boundaries, tolerance=1e-10, max_iterations=20000)

    assert np.allclose(convection_field.values, diffusion_field.values, atol=1e-6)


def test_constant_boundary_values_give_uniform_field():
    # With the same value on every boundary, the field is already uniform,
    # so it is a steady solution regardless of the velocity field.
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_convection_diffusion(
        mesh, (2.0, -1.0), _square_boundaries(left=4.0, right=4.0, top=4.0, bottom=4.0), diffusivity=1.0,
    )
    assert np.allclose(field.values, 4.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Maximum principle: the upwind + central-difference stencil produces a
# diagonally dominant matrix with non-positive off-diagonal entries, so the
# interior solution can never exceed the boundary extremes, whatever the
# velocity field is.
# ---------------------------------------------------------------------------

def test_non_uniform_boundaries_respect_maximum_principle():
    mesh = build_structured_mesh(nx=7, ny=7)
    boundaries = _square_boundaries(left=0.0, right=20.0, top=5.0, bottom=5.0)
    field = solve_convection_diffusion(mesh, (3.0, -2.0), boundaries, diffusivity=0.5)

    assert field.values.min() >= -1e-6
    assert field.values.max() <= 20.0 + 1e-6


def test_non_uniform_boundaries_are_applied_exactly():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = _square_boundaries(left=2.0, right=8.0, top=4.0, bottom=6.0)
    field = solve_convection_diffusion(mesh, (1.0, 0.5), boundaries, diffusivity=1.0)

    coords = mesh.cell_centers
    left_indices = np.where(np.isclose(coords[:, 0], coords[:, 0].min()))[0]
    right_indices = np.where(np.isclose(coords[:, 0], coords[:, 0].max()))[0]

    assert np.allclose(field.values[left_indices], 2.0)
    assert np.allclose(field.values[right_indices], 8.0)


# ---------------------------------------------------------------------------
# Velocity representations
# ---------------------------------------------------------------------------

def test_callable_velocity_matches_equivalent_constant_velocity():
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=1.0, right=9.0, top=4.0, bottom=6.0)

    constant_field = solve_convection_diffusion(mesh, (2.0, 1.0), boundaries, diffusivity=1.0)
    callable_field = solve_convection_diffusion(
        mesh, lambda x, y: (2.0, 1.0), boundaries, diffusivity=1.0,
    )

    assert np.allclose(constant_field.values, callable_field.values, atol=1e-10)


def test_array_velocity_matches_equivalent_constant_velocity():
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=1.0, right=9.0, top=4.0, bottom=6.0)

    constant_field = solve_convection_diffusion(mesh, (1.5, -0.5), boundaries, diffusivity=1.0)
    array_velocity = np.tile([1.5, -0.5], (mesh.n_cells, 1))
    array_field = solve_convection_diffusion(mesh, array_velocity, boundaries, diffusivity=1.0)

    assert np.allclose(constant_field.values, array_field.values, atol=1e-10)


def test_vector_field_velocity_matches_equivalent_constant_velocity():
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=1.0, right=9.0, top=4.0, bottom=6.0)

    constant_field = solve_convection_diffusion(mesh, (0.5, 0.5), boundaries, diffusivity=1.0)
    velocity_field = VectorField(mesh, np.tile([0.5, 0.5], (mesh.n_cells, 1)))
    vector_field_result = solve_convection_diffusion(mesh, velocity_field, boundaries, diffusivity=1.0)

    assert np.allclose(constant_field.values, vector_field_result.values, atol=1e-10)


def test_velocity_field_on_different_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    velocity_field = VectorField(other_mesh, np.zeros((other_mesh.n_cells, 2)))
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, velocity_field, _square_boundaries(), 1.0)


def test_velocity_field_with_wrong_component_count_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    velocity_field = VectorField(mesh, np.zeros((mesh.n_cells, 3)))
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, velocity_field, _square_boundaries(), 1.0)


def test_wrong_shape_velocity_array_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, np.zeros((mesh.n_cells, 3)), _square_boundaries(), 1.0)


def test_wrong_length_constant_velocity_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0, 0.0), _square_boundaries(), 1.0)


def test_callable_velocity_with_bad_return_value_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, lambda x, y: 1.0, _square_boundaries(), 1.0)


def test_invalid_velocity_type_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_convection_diffusion_system(mesh, "east", _square_boundaries(), 1.0)


# ---------------------------------------------------------------------------
# Solver methods
# ---------------------------------------------------------------------------

def test_jacobi_and_gauss_seidel_agree_on_solution():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = _square_boundaries(left=0.0, right=10.0, top=5.0, bottom=5.0)

    field_jacobi = solve_convection_diffusion(
        mesh, (1.0, 0.5), boundaries, diffusivity=1.0, method="jacobi", tolerance=1e-10, max_iterations=20000,
    )
    field_gauss_seidel = solve_convection_diffusion(
        mesh, (1.0, 0.5), boundaries, diffusivity=1.0, method="gauss_seidel",
        tolerance=1e-10, max_iterations=20000,
    )

    assert np.allclose(field_jacobi.values, field_gauss_seidel.values, atol=1e-4)


def test_default_method_is_gauss_seidel():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = _square_boundaries(left=1.0, right=2.0, top=3.0, bottom=4.0)

    default_field = solve_convection_diffusion(
        mesh, (1.0, 0.0), boundaries, diffusivity=1.0, tolerance=1e-10, max_iterations=20000,
    )
    explicit_field = solve_convection_diffusion(
        mesh, (1.0, 0.0), boundaries, diffusivity=1.0, method="gauss_seidel",
        tolerance=1e-10, max_iterations=20000,
    )

    assert np.allclose(default_field.values, explicit_field.values, atol=1e-12)


def test_unknown_solver_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_convection_diffusion(mesh, (1.0, 0.0), _square_boundaries(), diffusivity=1.0, method="fancy")


def test_solvers_converge_within_tolerance():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = assemble_convection_diffusion_system(
        mesh, (1.0, -1.0), _square_boundaries(left=1.0, right=3.0, top=2.0, bottom=4.0), 1.0,
    )

    solver = JacobiSolver(max_iterations=5000, tolerance=1e-8)
    solver.solve(system)

    assert solver.converged is True
    assert solver.residual_history[-1] < 1e-8


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def test_assemble_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        assemble_convection_diffusion_system("not a mesh", (1.0, 0.0), _square_boundaries(), 1.0)


def test_assemble_rejects_3d_mesh():
    mesh_3d = Mesh(
        cell_centers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_centers=np.array([[0.5, 0.0, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh_3d, (1.0, 0.0), _square_boundaries(), 1.0)


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
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), _square_boundaries(), 1.0)


# ---------------------------------------------------------------------------
# Boundary condition validation
# ---------------------------------------------------------------------------

def test_missing_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    incomplete = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), incomplete, 1.0)


def test_duplicate_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    duplicated = [
        FixedValueBC("left", 0.0),
        FixedValueBC("left", 1.0),
        FixedValueBC("top", 0.0),
        FixedValueBC("bottom", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), duplicated, 1.0)


def test_non_fixed_value_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
        ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), boundaries, 1.0)


def test_wrong_number_of_boundaries_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), _square_boundaries()[:3], 1.0)


def test_boundary_conditions_must_be_list_or_tuple():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), FixedValueBC("left", 0.0), 1.0)


# ---------------------------------------------------------------------------
# diffusivity validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_diffusivity", [True, "1.0", None])
def test_non_numeric_diffusivity_raises_type_error(bad_diffusivity):
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), _square_boundaries(), bad_diffusivity)


def test_non_positive_diffusivity_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_convection_diffusion_system(mesh, (1.0, 0.0), _square_boundaries(), 0.0)


# ---------------------------------------------------------------------------
# Equation description
# ---------------------------------------------------------------------------

def test_convection_diffusion_equation_returns_equation_instance():
    equation = convection_diffusion_equation()
    assert isinstance(equation, Equation)
    assert equation.lhs == "convection(phi)"
    assert equation.rhs == "diffusivity * laplacian(phi)"
    assert str(equation) == "convection(phi) = diffusivity * laplacian(phi)"
