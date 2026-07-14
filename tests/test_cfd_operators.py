import numpy as np
import pytest

from src.cfd import Mesh, ScalarField, VectorField, gradient, laplacian, divergence


@pytest.fixture
def mesh_2d():
    centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )
    return Mesh(
        cell_centers=centers,
        face_centers=np.array([[0.5, 0.0], [1.5, 0.0], [0.5, 1.0], [1.5, 1.0], [0.5, 2.0], [1.5, 2.0]]),
        face_areas=np.ones(6),
        cell_volumes=np.ones(9),
        owner_cells=np.array([0, 1, 3, 4, 6, 7]),
        neighbour_cells=np.array([1, 2, 4, 5, 7, 8]),
    )


def test_constant_field_gradient_and_laplacian(mesh_2d):
    field = ScalarField(mesh_2d, np.ones(mesh_2d.n_cells))

    grad = gradient(field)
    lap = laplacian(field)

    assert isinstance(grad, VectorField)
    assert isinstance(lap, ScalarField)
    assert np.allclose(grad.values, 0.0)
    assert np.allclose(lap.values, 0.0)


def test_linear_field_gradient(mesh_2d):
    x = mesh_2d.cell_centers[:, 0]
    y = mesh_2d.cell_centers[:, 1]
    values = x + 2.0 * y
    field = ScalarField(mesh_2d, values)

    grad = gradient(field)

    assert np.allclose(grad.values, np.column_stack([np.ones(mesh_2d.n_cells), 2.0 * np.ones(mesh_2d.n_cells)]))


def test_quadratic_field_laplacian(mesh_2d):
    x = mesh_2d.cell_centers[:, 0]
    y = mesh_2d.cell_centers[:, 1]
    values = x**2 + y**2
    field = ScalarField(mesh_2d, values)

    lap = laplacian(field)

    assert np.allclose(lap.values, 4.0)


def test_constant_vector_field_divergence(mesh_2d):
    values = np.column_stack([np.ones(mesh_2d.n_cells), 2.0 * np.ones(mesh_2d.n_cells)])
    field = VectorField(mesh_2d, values)

    divergence_field = divergence(field)

    assert isinstance(divergence_field, ScalarField)
    assert np.allclose(divergence_field.values, 0.0)


def test_invalid_inputs_raise_errors(mesh_2d):
    field = ScalarField(mesh_2d, np.ones(mesh_2d.n_cells))
    vector_field = VectorField(mesh_2d, np.column_stack([np.ones(mesh_2d.n_cells), np.ones(mesh_2d.n_cells)]))

    with pytest.raises(TypeError):
        gradient("not-a-field")

    with pytest.raises(TypeError):
        laplacian("not-a-field")

    with pytest.raises(TypeError):
        divergence("not-a-field")

    with pytest.raises(TypeError):
        divergence(field)

    wrong_mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0]]),
        face_centers=np.array([[0.5, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    bad_field = ScalarField(wrong_mesh, np.ones(wrong_mesh.n_cells))
    with pytest.raises(ValueError):
        gradient(bad_field)
