"""Simple boundary condition classes for structured CFD fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from .field import ScalarField
from .mesh import Mesh


class BoundaryCondition:
    """Base class for mesh boundary conditions."""

    def __init__(self, boundary: str) -> None:
        self.boundary = self._validate_boundary(boundary)

    @staticmethod
    def _validate_boundary(boundary: str) -> str:
        if boundary not in {"left", "right", "top", "bottom"}:
            raise ValueError("boundary must be one of: left, right, top, bottom")
        return boundary

    def apply(self, field: ScalarField) -> ScalarField:
        """Apply the boundary condition to a field."""
        raise NotImplementedError("BoundaryCondition subclasses must implement apply().")


class FixedValueBC(BoundaryCondition):
    """Set the selected boundary values to a fixed number."""

    def __init__(self, boundary: str, value: float) -> None:
        super().__init__(boundary)
        self.value = float(value)

    def apply(self, field: ScalarField) -> ScalarField:
        if not isinstance(field, ScalarField):
            raise TypeError("field must be a ScalarField instance.")
        self._validate_field_size(field)

        indices = self._boundary_indices(field.mesh)
        field.values[indices] = self.value
        return field

    def _validate_field_size(self, field: ScalarField) -> None:
        if field.values.shape[0] != field.mesh.n_cells:
            raise ValueError("field size is incompatible with the mesh.")

    def _boundary_indices(self, mesh: Mesh) -> np.ndarray:
        indices = []
        for index, (x, y) in enumerate(mesh.cell_centers):
            if self.boundary == "left" and np.isclose(x, np.min(mesh.cell_centers[:, 0])):
                indices.append(index)
            elif self.boundary == "right" and np.isclose(x, np.max(mesh.cell_centers[:, 0])):
                indices.append(index)
            elif self.boundary == "bottom" and np.isclose(y, np.min(mesh.cell_centers[:, 1])):
                indices.append(index)
            elif self.boundary == "top" and np.isclose(y, np.max(mesh.cell_centers[:, 1])):
                indices.append(index)
        return np.array(indices, dtype=int)


class ZeroGradientBC(BoundaryCondition):
    """Copy the nearest interior value to the boundary."""

    def apply(self, field: ScalarField) -> ScalarField:
        if not isinstance(field, ScalarField):
            raise TypeError("field must be a ScalarField instance.")
        self._validate_field_size(field)

        indices = self._boundary_indices(field.mesh)
        for index in indices:
            if self.boundary == "left":
                interior_index = self._find_adjacent_interior_index(field.mesh, index, axis=0, direction=1)
            elif self.boundary == "right":
                interior_index = self._find_adjacent_interior_index(field.mesh, index, axis=0, direction=-1)
            elif self.boundary == "bottom":
                interior_index = self._find_adjacent_interior_index(field.mesh, index, axis=1, direction=1)
            else:
                interior_index = self._find_adjacent_interior_index(field.mesh, index, axis=1, direction=-1)
            field.values[index] = field.values[interior_index]
        return field

    def _validate_field_size(self, field: ScalarField) -> None:
        if field.values.shape[0] != field.mesh.n_cells:
            raise ValueError("field size is incompatible with the mesh.")

    def _boundary_indices(self, mesh: Mesh) -> np.ndarray:
        indices = []
        for index, (x, y) in enumerate(mesh.cell_centers):
            if self.boundary == "left" and np.isclose(x, np.min(mesh.cell_centers[:, 0])):
                indices.append(index)
            elif self.boundary == "right" and np.isclose(x, np.max(mesh.cell_centers[:, 0])):
                indices.append(index)
            elif self.boundary == "bottom" and np.isclose(y, np.min(mesh.cell_centers[:, 1])):
                indices.append(index)
            elif self.boundary == "top" and np.isclose(y, np.max(mesh.cell_centers[:, 1])):
                indices.append(index)
        return np.array(indices, dtype=int)

    def _find_adjacent_interior_index(self, mesh: Mesh, boundary_index: int, axis: int, direction: int) -> int:
        point = mesh.cell_centers[boundary_index]
        for index, candidate in enumerate(mesh.cell_centers):
            if index == boundary_index:
                continue
            if axis == 0 and np.isclose(candidate[0], point[0] + direction * 1.0):
                return index
            if axis == 1 and np.isclose(candidate[1], point[1] + direction * 1.0):
                return index
        raise ValueError("could not find an adjacent interior cell for the requested boundary.")
