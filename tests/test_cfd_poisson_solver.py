import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ScalarField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.equation import Equation
from src.cfd.operators import laplacian
from src.cfd.poisson_solver import (
    assemble_poisson_system,
    poisson_equation,
    solve_poisson,
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
# Basic solving
# ---------------------------------------------------------------------------

def test_square_domain_solves_without_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_poisson(mesh, 0.0, _square_boundaries(left=1.0, right=1.0, top=1.0, bottom=1.0))
    assert field.values.shape[0] == mesh.n_cells


def test_square_domain_matches_mesh():
    mesh = build_structured_mesh(nx=6, ny=6)
    field = solve_poisson(mesh, 0.0, _square_boundaries())
    assert field.mesh is mesh


def test_zero_source_and_boundaries_reduces_to_laplace_solution():
    # With no source term and matching boundary values, Poisson reduces to
    # Laplace's equation, whose unique steady solution is that constant.
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_poisson(mesh, 0.0, _square_boundaries(left=10.0, right=10.0, top=10.0, bottom=10.0))
    assert np.allclose(field.values, 10.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Source term forms
# ---------------------------------------------------------------------------

def test_constant_source_term_accepted():
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_poisson(mesh, -2.0, _square_boundaries())
    assert field.values.shape[0] == mesh.n_cells


def test_callable_source_term_accepted():
    mesh = build_structured_mesh(nx=5, ny=5)
    field = solve_poisson(mesh, lambda x, y: x + y, _square_boundaries())
    assert field.values.shape[0] == mesh.n_cells


def test_array_source_term_accepted():
    mesh = build_structured_mesh(nx=5, ny=5)
    source = np.full(mesh.n_cells, 3.0)
    field = solve_poisson(mesh, source, _square_boundaries())
    assert field.values.shape[0] == mesh.n_cells


def test_scalar_field_source_term_accepted():
    mesh = build_structured_mesh(nx=5, ny=5)
    source_field = ScalarField(mesh, np.full(mesh.n_cells, 5.0))
    field = solve_poisson(mesh, source_field, _square_boundaries())
    assert field.values.shape[0] == mesh.n_cells


def test_scalar_field_source_term_on_different_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    source_field = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        assemble_poisson_system(mesh, source_field, _square_boundaries())


def test_wrong_length_array_source_term_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_poisson_system(mesh, np.zeros(3), _square_boundaries())


def test_bool_source_term_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_poisson_system(mesh, True, _square_boundaries())


def test_unsupported_source_term_type_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_poisson_system(mesh, "not a valid source", _square_boundaries())


# ---------------------------------------------------------------------------
# Manufactured solution: phi(x, y) = sin(pi*x) * sin(pi*y)
#
# This is zero along all four sides of the unit square (so a single
# FixedValueBC value of 0.0 per side is exact), and its Laplacian is the
# closed-form expression -2*pi^2*sin(pi*x)*sin(pi*y).
# ---------------------------------------------------------------------------

def _manufactured_source(x, y):
    return -2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def test_manufactured_solution_matches_analytical_field():
    mesh = build_structured_mesh(nx=21, ny=21)
    x, y = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]
    analytical = np.sin(np.pi * x) * np.sin(np.pi * y)

    boundaries = _square_boundaries()  # zero on all four sides

    field = solve_poisson(
        mesh, _manufactured_source, boundaries, method="gauss_seidel", max_iterations=20000, tolerance=1e-6
    )
    assert np.allclose(field.values, analytical, atol=5e-2)


def test_manufactured_solution_satisfies_discrete_laplacian():
    mesh = build_structured_mesh(nx=15, ny=15)
    boundaries = _square_boundaries()  # zero on all four sides
    field = solve_poisson(mesh, _manufactured_source, boundaries, max_iterations=20000, tolerance=1e-6)

    lap = laplacian(field)
    x, y = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]
    interior = (~np.isclose(x, x.min())) & (~np.isclose(x, x.max())) & (~np.isclose(y, y.min())) & (
        ~np.isclose(y, y.max())
    )
    expected = _manufactured_source(x[interior], y[interior])
    assert np.allclose(lap.values[interior], expected, atol=3e-1)


# ---------------------------------------------------------------------------
# Solver methods
# ---------------------------------------------------------------------------

def test_jacobi_and_gauss_seidel_agree_on_solution():
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=0.0, right=10.0, top=5.0, bottom=5.0)

    field_jacobi = solve_poisson(mesh, 1.0, boundaries, method="jacobi", tolerance=1e-10, max_iterations=20000)
    field_gauss_seidel = solve_poisson(
        mesh, 1.0, boundaries, method="gauss_seidel", tolerance=1e-10, max_iterations=20000
    )

    assert np.allclose(field_jacobi.values, field_gauss_seidel.values, atol=1e-4)


def test_default_method_is_gauss_seidel():
    mesh = build_structured_mesh(nx=6, ny=6)
    boundaries = _square_boundaries(left=1.0, right=2.0, top=3.0, bottom=4.0)
    system = assemble_poisson_system(mesh, 2.0, boundaries)

    default_field = solve_poisson(mesh, 2.0, boundaries, tolerance=1e-10, max_iterations=20000)
    gs_solution = GaussSeidelSolver(max_iterations=20000, tolerance=1e-10).solve(system)

    assert np.allclose(default_field.values, gs_solution, atol=1e-8)


def test_unknown_solver_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        solve_poisson(mesh, 0.0, _square_boundaries(), method="conjugate_gradient")


def test_solvers_converge_within_tolerance():
    mesh = build_structured_mesh(nx=5, ny=5)
    system = assemble_poisson_system(mesh, 1.0, _square_boundaries(left=1.0, right=3.0, top=2.0, bottom=4.0))

    solver = JacobiSolver(max_iterations=5000, tolerance=1e-8)
    solver.solve(system)

    assert solver.converged is True
    assert solver.residual_history[-1] < 1e-8


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def test_assemble_rejects_non_mesh_input():
    with pytest.raises(TypeError):
        assemble_poisson_system("not a mesh", 0.0, _square_boundaries())


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
        assemble_poisson_system(mesh_3d, 0.0, _square_boundaries())


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
        assemble_poisson_system(mesh, 0.0, _square_boundaries())


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
        assemble_poisson_system(mesh, 0.0, incomplete)


def test_duplicate_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    duplicated = [
        FixedValueBC("left", 0.0),
        FixedValueBC("left", 1.0),
        FixedValueBC("top", 0.0),
        FixedValueBC("bottom", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_poisson_system(mesh, 0.0, duplicated)


def test_non_fixed_value_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    boundaries = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
        ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        assemble_poisson_system(mesh, 0.0, boundaries)


def test_wrong_number_of_boundaries_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(ValueError):
        assemble_poisson_system(mesh, 0.0, _square_boundaries()[:3])


def test_boundary_conditions_must_be_list_or_tuple():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        assemble_poisson_system(mesh, 0.0, FixedValueBC("left", 0.0))


# ---------------------------------------------------------------------------
# Equation description
# ---------------------------------------------------------------------------

def test_poisson_equation_returns_equation_instance():
    equation = poisson_equation()
    assert isinstance(equation, Equation)
    assert equation.lhs == "laplacian(phi)"
    assert equation.rhs == "source_term"
    assert str(equation) == "laplacian(phi) = source_term"
