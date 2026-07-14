"""Finite-difference operators for structured Cartesian CFD meshes."""

from __future__ import annotations

import numpy as np

from .field import ScalarField, VectorField
from .mesh import Mesh


def gradient(field: ScalarField) -> VectorField:
    """Approximate the gradient of a scalar field on a uniform Cartesian mesh.

    Parameters
    ----------
    field : ScalarField
        Scalar field defined at cell centers.

    Returns
    -------
    VectorField
        Gradient components in each spatial dimension.
    """
    _validate_scalar_field(field)
    mesh = field.mesh
    _validate_mesh(mesh)

    coords = mesh.cell_centers
    values = field.values
    x_coords = np.unique(np.round(coords[:, 0], 12))
    y_coords = np.unique(np.round(coords[:, 1], 12))
    dx = float(np.diff(x_coords)[0])
    dy = float(np.diff(y_coords)[0])
    grid = _build_grid(values, coords)

    gradient_values = np.zeros((mesh.n_cells, coords.shape[1]), dtype=float)

    for index, (x, y) in enumerate(coords):
        ix = int(np.where(np.isclose(x_coords, x))[0][0])
        iy = int(np.where(np.isclose(y_coords, y))[0][0])
        gradient_values[index, 0] = _differentiate_along_axis(grid, ix, iy, axis=0, delta=dx)
        gradient_values[index, 1] = _differentiate_along_axis(grid, ix, iy, axis=1, delta=dy)

    return VectorField(mesh, gradient_values)


def laplacian(field: ScalarField) -> ScalarField:
    """Approximate the Laplacian of a scalar field on a uniform Cartesian mesh.

    Parameters
    ----------
    field : ScalarField
        Scalar field defined at cell centers.

    Returns
    -------
    ScalarField
        Laplacian values at each cell center.
    """
    _validate_scalar_field(field)
    mesh = field.mesh
    _validate_mesh(mesh)

    coords = mesh.cell_centers
    values = field.values
    x_coords = np.unique(np.round(coords[:, 0], 12))
    y_coords = np.unique(np.round(coords[:, 1], 12))
    dx = float(np.diff(x_coords)[0])
    dy = float(np.diff(y_coords)[0])
    grid = _build_grid(values, coords)

    laplacian_values = np.zeros(mesh.n_cells, dtype=float)

    for index, (x, y) in enumerate(coords):
        ix = int(np.where(np.isclose(x_coords, x))[0][0])
        iy = int(np.where(np.isclose(y_coords, y))[0][0])
        d2x = _second_derivative_along_axis(grid, ix, iy, axis=0, delta=dx)
        d2y = _second_derivative_along_axis(grid, ix, iy, axis=1, delta=dy)
        laplacian_values[index] = d2x + d2y

    return ScalarField(mesh, laplacian_values)


def divergence(vector_field: VectorField) -> ScalarField:
    """Approximate the divergence of a vector field on a uniform Cartesian mesh.

    Parameters
    ----------
    vector_field : VectorField
        Vector field defined at cell centers.

    Returns
    -------
    ScalarField
        Divergence values at each cell center.
    """
    if not isinstance(vector_field, VectorField):
        raise TypeError("divergence expects a VectorField instance.")

    mesh = vector_field.mesh
    _validate_mesh(mesh)

    coords = mesh.cell_centers
    values_x = vector_field.values[:, 0]
    values_y = vector_field.values[:, 1]
    x_coords = np.unique(np.round(coords[:, 0], 12))
    y_coords = np.unique(np.round(coords[:, 1], 12))
    dx = float(np.diff(x_coords)[0])
    dy = float(np.diff(y_coords)[0])
    grid_x = _build_grid(values_x, coords)
    grid_y = _build_grid(values_y, coords)

    divergence_values = np.zeros(mesh.n_cells, dtype=float)

    for index, (x, y) in enumerate(coords):
        ix = int(np.where(np.isclose(x_coords, x))[0][0])
        iy = int(np.where(np.isclose(y_coords, y))[0][0])
        dfx = _differentiate_along_axis(grid_x, ix, iy, axis=0, delta=dx)
        dfy = _differentiate_along_axis(grid_y, ix, iy, axis=1, delta=dy)
        divergence_values[index] = dfx + dfy

    return ScalarField(mesh, divergence_values)


def _validate_scalar_field(field: ScalarField) -> None:
    if not isinstance(field, ScalarField):
        raise TypeError("expected a ScalarField instance.")


def _validate_mesh(mesh: Mesh) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an instance of Mesh.")
    if mesh.cell_centers.shape[1] != 2:
        raise ValueError("this operator implementation supports only 2D Cartesian meshes.")
    if mesh.n_cells < 3:
        raise ValueError("mesh must contain at least three cells for finite-difference operators.")

    x_values = np.unique(np.round(mesh.cell_centers[:, 0], 12))
    y_values = np.unique(np.round(mesh.cell_centers[:, 1], 12))
    if x_values.size < 3 or y_values.size < 3:
        raise ValueError("mesh must contain at least three points along each axis.")

    dx = np.diff(x_values)
    dy = np.diff(y_values)
    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("mesh must be uniform in each direction.")


def _build_grid(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    x_coords = np.unique(np.round(coords[:, 0], 12))
    y_coords = np.unique(np.round(coords[:, 1], 12))
    grid = np.zeros((x_coords.size, y_coords.size), dtype=float)
    for index, (x, y) in enumerate(coords):
        ix = int(np.where(np.isclose(x_coords, x))[0][0])
        iy = int(np.where(np.isclose(y_coords, y))[0][0])
        grid[ix, iy] = values[index]
    return grid


def _differentiate_along_axis(grid: np.ndarray, ix: int, iy: int, axis: int, delta: float) -> float:
    nx, ny = grid.shape
    if axis == 0:
        if ix == 0:
            return (grid[1, iy] - grid[0, iy]) / delta
        if ix == nx - 1:
            return (grid[ix, iy] - grid[ix - 1, iy]) / delta
        return (grid[ix + 1, iy] - grid[ix - 1, iy]) / (2.0 * delta)

    if iy == 0:
        return (grid[ix, 1] - grid[ix, 0]) / delta
    if iy == ny - 1:
        return (grid[ix, iy] - grid[ix, iy - 1]) / delta
    return (grid[ix, iy + 1] - grid[ix, iy - 1]) / (2.0 * delta)


def _second_derivative_along_axis(grid: np.ndarray, ix: int, iy: int, axis: int, delta: float) -> float:
    nx, ny = grid.shape
    if axis == 0:
        if ix == 0:
            return (grid[0, iy] - 2.0 * grid[1, iy] + grid[2, iy]) / (delta ** 2)
        if ix == nx - 1:
            return (grid[ix - 2, iy] - 2.0 * grid[ix - 1, iy] + grid[ix, iy]) / (delta ** 2)
        return (grid[ix + 1, iy] - 2.0 * grid[ix, iy] + grid[ix - 1, iy]) / (delta ** 2)

    if iy == 0:
        return (grid[ix, 0] - 2.0 * grid[ix, 1] + grid[ix, 2]) / (delta ** 2)
    if iy == ny - 1:
        return (grid[ix, iy - 2] - 2.0 * grid[ix, iy - 1] + grid[ix, iy]) / (delta ** 2)
    return (grid[ix, iy + 1] - 2.0 * grid[ix, iy] + grid[ix, iy - 1]) / (delta ** 2)
