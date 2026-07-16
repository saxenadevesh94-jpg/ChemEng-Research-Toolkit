import numpy as np
import pytest

from src.cfd import FixedValueBC, Mesh, ScalarField, ZeroGradientBC
from src.cfd.diffusion_solver import build_structured_mesh
from src.cfd.equation import Equation
from src.cfd.solvers import GaussSeidelSolver, JacobiSolver
from src.cfd.transient_diffusion import (
    assemble_transient_diffusion_system,
    solve_transient_diffusion,
    transient_diffusion_equation,
)


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
    initial_field = ScalarField(mesh, np.full(mesh.n_cells, 1.0))
    field = solve_transient_diffusion(
        mesh, initial_field, _square_boundaries(left=1.0, right=1.0, top=1.0, bottom=1.0),
        diffusivity=1.0, dt=0.01, total_time=0.05,
    )
    assert field.values.shape[0] == mesh.n_cells


def test_square_domain_matches_mesh():
    mesh = build_structured_mesh(nx=6, ny=6)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    field = solve_transient_diffusion(
        mesh, initial_field, _square_boundaries(), diffusivity=1.0, dt=0.01, total_time=0.02,
    )
    assert field.mesh is mesh


def test_uniform_steady_state_field_is_unchanged():
    # A field already equal to its matching Dirichlet boundaries is a steady
    # solution of the diffusion equation, so time-stepping should not move it.
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.full(mesh.n_cells, 3.0))
    field = solve_transient_diffusion(
        mesh, initial_field, _square_boundaries(left=3.0, right=3.0, top=3.0, bottom=3.0),
        diffusivity=1.0, dt=0.05, total_time=0.5,
    )
    assert np.allclose(field.values, 3.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Manufactured solution: phi(x, y, t) = sin(pi*x) * sin(pi*y) * exp(-2*pi^2*alpha*t)
#
# This satisfies d(phi)/dt = alpha * laplacian(phi) exactly, and is zero
# along all four sides of the unit square at every time, so a single
# FixedValueBC value of 0.0 per side is exact for all t.
# ---------------------------------------------------------------------------

def _manufactured_solution(x, y, t, alpha):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2.0 * np.pi ** 2 * alpha * t)


def test_manufactured_solution_matches_analytical_decay():
    mesh = build_structured_mesh(nx=11, ny=11)
    x, y = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]
    alpha = 1.0
    total_time = 0.02

    initial_field = ScalarField(mesh, _manufactured_solution(x, y, 0.0, alpha))
    boundaries = _square_boundaries()  # zero on all four sides

    field = solve_transient_diffusion(
        mesh, initial_field, boundaries, diffusivity=alpha, dt=0.002, total_time=total_time,
        method="gauss_seidel", max_iterations=5000, tolerance=1e-8,
    )

    analytical = _manufactured_solution(x, y, total_time, alpha)
    assert np.allclose(field.values, analytical, atol=2e-2)


# ---------------------------------------------------------------------------
# Solver methods
# ---------------------------------------------------------------------------

def test_jacobi_and_gauss_seidel_agree_on_solution():
    mesh = build_structured_mesh(nx=6, ny=6)
    initial_field = ScalarField(mesh, np.full(mesh.n_cells, 2.0))
    boundaries = _square_boundaries(left=0.0, right=10.0, top=5.0, bottom=5.0)

    field_jacobi = solve_transient_diffusion(
        mesh, initial_field, boundaries, diffusivity=1.0, dt=0.01, total_time=0.05,
        method="jacobi", tolerance=1e-8, max_iterations=5000,
    )
    field_gauss_seidel = solve_transient_diffusion(
        mesh, initial_field, boundaries, diffusivity=1.0, dt=0.01, total_time=0.05,
        method="gauss_seidel", tolerance=1e-8, max_iterations=5000,
    )

    assert np.allclose(field_jacobi.values, field_gauss_seidel.values, atol=1e-4)


def test_default_method_is_gauss_seidel():
    mesh = build_structured_mesh(nx=6, ny=6)
    initial_field = ScalarField(mesh, np.full(mesh.n_cells, 1.0))
    boundaries = _square_boundaries(left=1.0, right=2.0, top=3.0, bottom=4.0)

    default_field = solve_transient_diffusion(
        mesh, initial_field, boundaries, diffusivity=1.0, dt=0.01, total_time=0.02,
        tolerance=1e-10, max_iterations=20000,
    )
    explicit_field = solve_transient_diffusion(
        mesh, initial_field, boundaries, diffusivity=1.0, dt=0.01, total_time=0.02,
        method="gauss_seidel", tolerance=1e-10, max_iterations=20000,
    )

    assert np.allclose(default_field.values, explicit_field.values, atol=1e-12)


def test_unknown_solver_method_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=1.0, dt=0.01, total_time=0.01,
            method="conjugate_gradient",
        )


def test_solvers_converge_within_tolerance():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.full(mesh.n_cells, 1.0))
    system = assemble_transient_diffusion_system(
        mesh, initial_field, _square_boundaries(left=1.0, right=3.0, top=2.0, bottom=4.0),
        diffusivity=1.0, dt=0.01,
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
        assemble_transient_diffusion_system("not a mesh", None, _square_boundaries(), 1.0, 0.01)


def test_assemble_rejects_3d_mesh():
    mesh_3d = Mesh(
        cell_centers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_centers=np.array([[0.5, 0.0, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    initial_field = ScalarField(mesh_3d, np.zeros(mesh_3d.n_cells))
    with pytest.raises(ValueError):
        assemble_transient_diffusion_system(mesh_3d, initial_field, _square_boundaries(), 1.0, 0.01)


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
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        assemble_transient_diffusion_system(mesh, initial_field, _square_boundaries(), 1.0, 0.01)


# ---------------------------------------------------------------------------
# Boundary condition validation
# ---------------------------------------------------------------------------

def test_missing_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    incomplete = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_transient_diffusion_system(mesh, initial_field, incomplete, 1.0, 0.01)


def test_duplicate_boundary_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    duplicated = [
        FixedValueBC("left", 0.0),
        FixedValueBC("left", 1.0),
        FixedValueBC("top", 0.0),
        FixedValueBC("bottom", 0.0),
    ]
    with pytest.raises(ValueError):
        assemble_transient_diffusion_system(mesh, initial_field, duplicated, 1.0, 0.01)


def test_non_fixed_value_bc_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    boundaries = [
        FixedValueBC("left", 0.0),
        FixedValueBC("right", 0.0),
        FixedValueBC("top", 0.0),
        ZeroGradientBC("bottom"),
    ]
    with pytest.raises(TypeError):
        assemble_transient_diffusion_system(mesh, initial_field, boundaries, 1.0, 0.01)


def test_wrong_number_of_boundaries_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        assemble_transient_diffusion_system(mesh, initial_field, _square_boundaries()[:3], 1.0, 0.01)


def test_boundary_conditions_must_be_list_or_tuple():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(TypeError):
        assemble_transient_diffusion_system(mesh, initial_field, FixedValueBC("left", 0.0), 1.0, 0.01)


# ---------------------------------------------------------------------------
# initial_field validation
# ---------------------------------------------------------------------------

def test_non_scalar_field_initial_field_raises_type_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    with pytest.raises(TypeError):
        solve_transient_diffusion(
            mesh, np.zeros(mesh.n_cells), _square_boundaries(), diffusivity=1.0, dt=0.01, total_time=0.01,
        )


def test_initial_field_on_different_mesh_raises_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    other_mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(other_mesh, np.zeros(other_mesh.n_cells))
    with pytest.raises(ValueError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=1.0, dt=0.01, total_time=0.01,
        )


# ---------------------------------------------------------------------------
# diffusivity / dt / total_time validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_diffusivity", [True, "1.0", None])
def test_non_numeric_diffusivity_raises_type_error(bad_diffusivity):
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(TypeError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=bad_diffusivity, dt=0.01, total_time=0.01,
        )


def test_non_positive_diffusivity_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=0.0, dt=0.01, total_time=0.01,
        )


def test_non_positive_dt_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=1.0, dt=-0.01, total_time=0.01,
        )


def test_non_positive_total_time_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=1.0, dt=0.01, total_time=0.0,
        )


def test_total_time_not_multiple_of_dt_raises_value_error():
    mesh = build_structured_mesh(nx=5, ny=5)
    initial_field = ScalarField(mesh, np.zeros(mesh.n_cells))
    with pytest.raises(ValueError):
        solve_transient_diffusion(
            mesh, initial_field, _square_boundaries(), diffusivity=1.0, dt=0.03, total_time=0.1,
        )


# ---------------------------------------------------------------------------
# Equation description
# ---------------------------------------------------------------------------

def test_transient_diffusion_equation_returns_equation_instance():
    equation = transient_diffusion_equation()
    assert isinstance(equation, Equation)
    assert equation.lhs == "ddt(phi)"
    assert equation.rhs == "diffusivity * laplacian(phi)"
    assert str(equation) == "ddt(phi) = diffusivity * laplacian(phi)"
