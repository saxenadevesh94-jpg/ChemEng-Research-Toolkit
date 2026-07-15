import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ZeroGradientBC
from src.cfd.diffusion_solver import (
    assemble_diffusion_system,
    build_structured_mesh,
    solve_diffusion,
)
from src.cfd.solvers import GaussSeidelSolver, JacobiSolver


def _square_boundaries(left=0.0, right=0.0, top=0.0, bottom=0.0):
    return [
        FixedValueBC("left", left),
        FixedValueBC("right", right),
        FixedValueBC("top", top),
        FixedValueBC("bottom", bottom),
    ]


# ---------------------------------------------------------------------------
# Square domain
# ---------------------------------------------------------------------------

def test_square_domain_solves_without_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_diffusion(mesh, _square_boundaries(left=1.0, right=1.0, top=1.0, bottom=1.0))
    assert field.values.shape[0] == mesh.n_cells


def test_square_domain_matches_mesh():
    mesh = build_structured_mesh(nx=6, ny=6)
    field = solve_diffusion(mesh, _square_boundaries())
    assert field.mesh is mesh


# ---------------------------------------------------------------------------
# Constant boundary values
# ---------------------------------------------------------------------------

def test_constant_boundary_values_give_uniform_field():
    # With Laplacian(phi) = 0 and the same value on every boundary, the
    # unique steady-state solution is that constant value everywhere.
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_diffusion(mesh, _square_boundaries(left=10.0, right=10.0, top=10.0, bottom=10.0))
    assert np.allclose(field.values, 10.0, atol=1e-4)


def test_zero_boundary_values_give_zero_field():
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_diffusion(mesh, _square_boundaries())
    assert np.allclose(field.values, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Non-uniform boundary values
# ---------------------------------------------------------------------------

def test_non_uniform_boundaries_respect_maximum_principle():
    # A harmonic function's interior values never exceed the max or fall
    # below the min of its boundary values.
    mesh = build_structured_mesh(nx=7, ny=7)
    boundaries = _square_boundaries(left=0.0, right=20.0, top=5.0, bottom=5.0)
    field = solve_diffusion(mesh, boundaries)

    assert field.values.min() >= -1e-6
    assert field.values.max() <= 20.0 + 1e-6


def test_non_uniform_boundaries_are_applied_exactly():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = _square_boundaries(left=2.0, right=8.0, top=4.0, bottom=6.0)
    field = solve_diffusion(mesh, boundaries)

    coords = mesh.cell_centers
    left_indices = np.where(np.isclose(coords[:, 0], coords[:, 0].min()))[0]
    right_indices = np.where(np.isclose(coords[:, 0], coords[:, 0].max()))[0]

    assert np.allclose(field.values[left_indices], 2.0)
    assert np.allclose(field.values[right_indices], 8.0)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def test_jacobi_solver_converges_within_tolerance():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = assemble_diffusion_system(mesh, _square_boundaries(left=1.0, right=3.0, top=2.0, bottom=4.0))

    solver = JacobiSolver(max_iterations=5000, tolerance=1e-8)
    solver.solve(system)

    assert solver.converged is True
    assert solver.residual_history[-1] < 1e-8


def test_gauss_seidel_converges_in_fewer_or_equal_iterations_than_jacobi():
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=1.0, right=3.0, top=2.0, bottom=4.0)

    jacobi_system = assemble_diffusion_system(mesh, boundaries)
    gauss_seidel_system = assemble_diffusion_system(mesh, boundaries)

    jacobi = JacobiSolver(max_iterations=10000, tolerance=1e-8)
    gauss_seidel = GaussSeidelSolver(max_iterations=10000, tolerance=1e-8)

    jacobi.solve(jacobi_system)
    gauss_seidel.solve(gauss_seidel_system)

    assert jacobi.converged is True
    assert gauss_seidel.converged is True
    # Gauss-Seidel uses fresher values mid-sweep, so it should not need more
    # iterations than Jacobi to reach the same tolerance.
    assert gauss_seidel.iterations_run <= jacobi.iterations_run


def test_jacobi_and_gauss_seidel_agree_on_solution():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = _square_boundaries(left=0.0, right=10.0, top=5.0, bottom=5.0)

    field_jacobi = solve_diffusion(mesh, boundaries, method="jacobi", tolerance=1e-10, max_iterations=20000)
    field_gauss_seidel = solve_diffusion(
        mesh, boundaries, method="gauss_seidel", tolerance=1e-10, max_iterations=20000
    )

    assert np.allclose(field_jacobi.values, field_gauss_seidel.values, atol=1e-4)


def test_unknown_solver_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_diffusion(mesh, _square_boundaries(), method="conjugate_gradient")


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def test_mesh_too_small_raises_error():
    with pytest.raises(ValueError):
        build_structured_mesh(nx=2, ny=5)

    with pytest.raises(ValueError):
        build_structured_mesh(nx=5, ny=2)


def test_non_integer_mesh_size_raises_error():
    with pytest.raises(TypeError):
        build_structured_mesh(nx=5.0, ny=5)


def test_assemble_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        assemble_diffusion_system("not a mesh", _square_boundaries())


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
        assemble_diffusion_system(mesh_3d, _square_boundaries())


def test_assemble_rejects_non_uniform_spacing():
    # x-coordinates are irregularly spaced (0, 1, 3) instead of uniform.
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
        assemble_diffusion_system(mesh, _square_boundaries())


# ---------------------------------------------------------------------------
# Invalid boundary conditions
# ---------------------------------------------------------------------------

def test_missing_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    incomplete = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_diffusion_system(mesh, incomplete)


def test_duplicate_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    duplicated = [
        FixedValueBC("left", 0.0),
        FixedValueBC("left", 1.0),
        FixedValueBC("top", 0.0),
        FixedValueBC("bottom", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_diffusion_system(mesh, duplicated)


def test_non_fixed_value_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
        ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        assemble_diffusion_system(mesh, boundaries)


def test_wrong_number_of_boundaries_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_diffusion_system(mesh, _square_boundaries()[:3])


def test_boundary_conditions_must_be_list_or_tuple():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_diffusion_system(mesh, FixedValueBC("left", 0.0))
