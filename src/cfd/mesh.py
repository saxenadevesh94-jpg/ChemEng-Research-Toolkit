"""Simple mesh container for CFD-style data."""

from __future__ import annotations

from typing import Optional

import numpy as np


class Mesh:
    """Store geometric data for a simple CFD mesh.

    Parameters
    ----------
    cell_centers : numpy.ndarray
        Array of shape ``(n_cells, 3)`` or ``(n_cells, 2)`` with cell centers.
    face_centers : numpy.ndarray
        Array of shape ``(n_faces, 3)`` or ``(n_faces, 2)`` with face centers.
    face_areas : numpy.ndarray
        Array of shape ``(n_faces,)`` with face areas.
    cell_volumes : numpy.ndarray
        Array of shape ``(n_cells,)`` with cell volumes.
    owner_cells : numpy.ndarray
        Array of shape ``(n_faces,)`` with owner cell indices.
    neighbour_cells : numpy.ndarray
        Array of shape ``(n_faces,)`` with neighbour cell indices.
    """

    def __init__(
        self,
        cell_centers: np.ndarray,
        face_centers: np.ndarray,
        face_areas: np.ndarray,
        cell_volumes: np.ndarray,
        owner_cells: np.ndarray,
        neighbour_cells: np.ndarray,
    ) -> None:
        self.cell_centers = self._validate_array(cell_centers, "cell_centers")
        self.face_centers = self._validate_array(face_centers, "face_centers")
        self.face_areas = self._validate_array(face_areas, "face_areas")
        self.cell_volumes = self._validate_array(cell_volumes, "cell_volumes")
        self.owner_cells = self._validate_array(owner_cells, "owner_cells")
        self.neighbour_cells = self._validate_array(neighbour_cells, "neighbour_cells")

        if self.cell_centers.ndim != 2:
            raise ValueError("cell_centers must be a 2D array.")
        if self.face_centers.ndim != 2:
            raise ValueError("face_centers must be a 2D array.")
        if self.face_areas.ndim != 1:
            raise ValueError("face_areas must be a 1D array.")
        if self.cell_volumes.ndim != 1:
            raise ValueError("cell_volumes must be a 1D array.")
        if self.owner_cells.ndim != 1:
            raise ValueError("owner_cells must be a 1D array.")
        if self.neighbour_cells.ndim != 1:
            raise ValueError("neighbour_cells must be a 1D array.")

        if self.cell_centers.shape[0] != self.cell_volumes.shape[0]:
            raise ValueError("cell_volumes length must match the number of cells.")
        if self.face_centers.shape[0] != self.face_areas.shape[0]:
            raise ValueError("face_areas length must match the number of faces.")
        if self.owner_cells.shape[0] != self.face_areas.shape[0]:
            raise ValueError("owner_cells length must match the number of faces.")
        if self.neighbour_cells.shape[0] != self.face_areas.shape[0]:
            raise ValueError("neighbour_cells length must match the number of faces.")

        if self.cell_centers.shape[1] != self.face_centers.shape[1]:
            raise ValueError("cell_centers and face_centers must have the same dimension.")

        self.n_cells = self.cell_centers.shape[0]
        self.n_faces = self.face_centers.shape[0]

    @staticmethod
    def _validate_array(values: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
        if values.ndim == 0:
            raise ValueError(f"{name} must be an array with at least one dimension.")
        return values
