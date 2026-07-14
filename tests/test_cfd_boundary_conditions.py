import numpy as np
import pytest

from src.cfd import Mesh, ScalarField, FixedValueBC, ZeroGradientBC


@pytest.fixture
def mesh_2d():
    cell_centers = np.array(
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
        cell_centers=cell_centers,
        face_centers=np.array([[0.5, 0.0], [1.5, 0.0], [0.5, 1.0], [1.5, 1.0], [0.5, 2.0], [1.5, 2.0]]),
        face_areas=np.ones(6),
        cell_volumes=np.ones(9),
        owner_cells=np.array([0, 1, 3, 4, 6, 7]),
        neighbour_cells=np.array([1, 2, 4, 5, 7, 8]),
    )


def test_fixed_value_bc_sets_boundary_correctly(mesh_2d):
    values = np.arange(1.0, 10.0)
    field = ScalarField(mesh_2d, values)

    bc = FixedValueBC("left", 7.0)
    bc.apply(field)

    assert np.allclose(field.values[[0, 3, 6]], 7.0)


def test_zero_gradient_bc_copies_adjacent_interior_values(mesh_2d):
    values = np.arange(1.0, 10.0)
    field = ScalarField(mesh_2d, values)

    bc = ZeroGradientBC("left")
    bc.apply(field)

    assert np.allclose(field.values[[0, 3, 6]], [2.0, 5.0, 8.0])


def test_invalid_boundary_names_raise_value_error():
    with pytest.raises(ValueError):
        FixedValueBC("diagonal", 1.0)

    with pytest.raises(ValueError):
        ZeroGradientBC("diagonal")


def test_incompatible_field_mesh_raises_error(mesh_2d):
    other_mesh = Mesh(
        cell_centers=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_centers=np.array([[0.5, 0.0]]),
        face_areas=np.array([1.0]),
        cell_volumes=np.array([1.0, 1.0, 1.0]),
        owner_cells=np.array([0]),
        neighbour_cells=np.array([1]),
    )
    field = ScalarField(other_mesh, np.ones(other_mesh.n_cells))

    bc = FixedValueBC("left", 3.0)
    with pytest.raises(ValueError):
        bc.apply(field)
