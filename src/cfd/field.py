"""Scalar and vector field classes attached to a CFD mesh."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .mesh import Mesh


class _BaseField:
    """Base class for mesh-backed fields."""

    def __init__(self, mesh: Mesh, values: np.ndarray) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be an instance of Mesh.")

        self.mesh = mesh
        self.values = self._validate_values(values)

    def _validate_values(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            raise ValueError("Field values must be an array with at least one dimension.")
        return array

    def copy(self) -> "_BaseField":
        """Return a copy of the field with the same values."""
        return self.__class__(self.mesh, self.values.copy())

    def fill(self, value: float) -> None:
        """Fill the field with a single numeric value."""
        self.values = np.full_like(self.values, value, dtype=float)

    def min(self) -> float:
        """Return the minimum value in the field."""
        return float(np.min(self.values))

    def max(self) -> float:
        """Return the maximum value in the field."""
        return float(np.max(self.values))

    def mean(self) -> float:
        """Return the mean value in the field."""
        return float(np.mean(self.values))


class ScalarField(_BaseField):
    """Store a scalar value at each cell of the mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh to attach the field to.
    values : numpy.ndarray
        Array of shape ``(n_cells,)`` with one scalar value per cell.
    """

    def _validate_values(self, values: np.ndarray) -> np.ndarray:
        array = super()._validate_values(values)
        if array.ndim != 1:
            raise ValueError("ScalarField values must be a 1D array.")
        if array.shape[0] != self.mesh.n_cells:
            raise ValueError("ScalarField values length must match the number of cells.")
        return array


class VectorField(_BaseField):
    """Store a vector value at each cell of the mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh to attach the field to.
    values : numpy.ndarray
        Array of shape ``(n_cells, n_components)`` with one vector per cell.
    """

    def _validate_values(self, values: np.ndarray) -> np.ndarray:
        array = super()._validate_values(values)
        if array.ndim != 2:
            raise ValueError("VectorField values must be a 2D array.")
        if array.shape[0] != self.mesh.n_cells:
            raise ValueError("VectorField values length must match the number of cells.")
        if array.shape[1] < 1:
            raise ValueError("VectorField values must contain at least one component.")
        return array
