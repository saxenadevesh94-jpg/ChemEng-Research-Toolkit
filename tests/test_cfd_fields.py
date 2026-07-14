import numpy as np
import pytest

from src.cfd import Mesh, ScalarField, VectorField


def test_mesh_creation():
    cell_centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    face_centers = np.array([[0.5, 0.0], [0.5, 0.5]])
    face_areas = np.array([1.0, 1.0])
    cell_volumes = np.array([0.5, 0.5, 0.5])
    owner_cells = np.array([0, 1])
    neighbour_cells = np.array([1, 2])

    mesh = Mesh(
        cell_centers=cell_centers,
        face_centers=face_centers,
        face_areas=face_areas,
        cell_volumes=cell_volumes,
        owner_cells=owner_cells,
        neighbour_cells=neighbour_cells,
    )

    assert mesh.n_cells == 3
    assert mesh.n_faces == 2
    assert np.array_equal(mesh.cell_centers, cell_centers)
    assert np.array_equal(mesh.face_areas, face_areas)


def test_scalar_field_creation():
    mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_centers=np.array([[0.5, 0.0], [0.5, 0.5]]),
        face_areas=np.array([1.0, 1.0]),
        cell_volumes=np.array([0.5, 0.5, 0.5]),
        owner_cells=np.array([0, 1]),
        neighbour_cells=np.array([1, 2]),
    )
    values = np.array([1.0, 2.0, 3.0])

    field = ScalarField(mesh, values)

    assert field.mesh is mesh
    assert np.array_equal(field.values, values)


def test_vector_field_creation():
    mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_centers=np.array([[0.5, 0.0], [0.5, 0.5]]),
        face_areas=np.array([1.0, 1.0]),
        cell_volumes=np.array([0.5, 0.5, 0.5]),
        owner_cells=np.array([0, 1]),
        neighbour_cells=np.array([1, 2]),
    )
    values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    field = VectorField(mesh, values)

    assert field.mesh is mesh
    assert np.array_equal(field.values, values)


def test_invalid_field_dimensions_raise_value_error():
    mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_centers=np.array([[0.5, 0.0], [0.5, 0.5]]),
        face_areas=np.array([1.0, 1.0]),
        cell_volumes=np.array([0.5, 0.5, 0.5]),
        owner_cells=np.array([0, 1]),
        neighbour_cells=np.array([1, 2]),
    )

    with pytest.raises(ValueError):
        ScalarField(mesh, np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        VectorField(mesh, np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_scalar_field_fill_and_copy():
    mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_centers=np.array([[0.5, 0.0], [0.5, 0.5]]),
        face_areas=np.array([1.0, 1.0]),
        cell_volumes=np.array([0.5, 0.5, 0.5]),
        owner_cells=np.array([0, 1]),
        neighbour_cells=np.array([1, 2]),
    )
    field = ScalarField(mesh, np.array([1.0, 2.0, 3.0]))

    field.fill(4.0)
    assert np.all(field.values == 4.0)

    copied = field.copy()
    assert copied is not field
    assert np.array_equal(copied.values, field.values)
    assert copied.mesh is mesh


def test_field_statistics_methods():
    mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_centers=np.array([[0.5, 0.0], [0.5, 0.5]]),
        face_areas=np.array([1.0, 1.0]),
        cell_volumes=np.array([0.5, 0.5, 0.5]),
        owner_cells=np.array([0, 1]),
        neighbour_cells=np.array([1, 2]),
    )
    scalar_field = ScalarField(mesh, np.array([1.0, 2.0, 3.0]))
    vector_field = VectorField(mesh, np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]))

    assert scalar_field.min() == 1.0
    assert scalar_field.max() == 3.0
    assert scalar_field.mean() == 2.0
    assert vector_field.min() == 0.0
    assert vector_field.max() == 3.0
    assert vector_field.mean() == 1.5
