"""Lightweight sparse matrix and linear system containers for CFD assembly."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class SparseMatrix:
    """Store matrix coefficients for a square system in a dictionary of keys format.

    Only non-zero entries are kept in memory. This is intentionally simple —
    no compression scheme is applied. A future assembly step can convert the
    result to a proper sparse format before handing it to a solver.

    Parameters
    ----------
    size : int
        Number of rows (and columns) in the square matrix.

    Raises
    ------
    TypeError
        If ``size`` is not an integer.
    ValueError
        If ``size`` is less than one.

    Examples
    --------
    >>> mat = SparseMatrix(size=3)
    >>> mat.set(0, 0, 4.0)
    >>> mat.set(0, 1, -1.0)
    >>> mat.get(0, 1)
    -1.0
    >>> mat.to_array()
    array([[ 4., -1.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    """

    def __init__(self, size: int) -> None:
        if not isinstance(size, int):
            raise TypeError(f"size must be an integer, got {type(size).__name__}.")
        if size < 1:
            raise ValueError("size must be at least 1.")
        self._size = size
        # Keys are (row, col) tuples; values are floats.
        self._data: Dict[Tuple[int, int], float] = {}

    @property
    def size(self) -> int:
        """Number of rows and columns in the matrix."""
        return self._size

    def _validate_index(self, row: int, col: int) -> None:
        """Raise an error when row or col fall outside [0, size)."""
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("row and col must be integers.")
        if not (0 <= row < self._size):
            raise ValueError(f"row {row} is out of range for a {self._size}x{self._size} matrix.")
        if not (0 <= col < self._size):
            raise ValueError(f"col {col} is out of range for a {self._size}x{self._size} matrix.")

    def set(self, row: int, col: int, value: float) -> None:
        """Set the coefficient at position ``(row, col)``.

        Parameters
        ----------
        row : int
            Row index (zero-based).
        col : int
            Column index (zero-based).
        value : float
            Coefficient value.

        Raises
        ------
        TypeError
            If ``row`` or ``col`` are not integers.
        ValueError
            If ``row`` or ``col`` are out of range.
        """
        self._validate_index(row, col)
        self._data[(row, col)] = float(value)

    def get(self, row: int, col: int) -> float:
        """Return the coefficient at position ``(row, col)``.

        Parameters
        ----------
        row : int
            Row index (zero-based).
        col : int
            Column index (zero-based).

        Returns
        -------
        float
            Stored coefficient, or ``0.0`` if no entry exists at that position.

        Raises
        ------
        TypeError
            If ``row`` or ``col`` are not integers.
        ValueError
            If ``row`` or ``col`` are out of range.
        """
        self._validate_index(row, col)
        return self._data.get((row, col), 0.0)

    def to_array(self) -> np.ndarray:
        """Convert the sparse matrix to a dense NumPy array.

        Returns
        -------
        numpy.ndarray
            Dense array of shape ``(size, size)``.
        """
        dense = np.zeros((self._size, self._size), dtype=float)
        for (row, col), value in self._data.items():
            dense[row, col] = value
        return dense

    def __repr__(self) -> str:
        return f"SparseMatrix(size={self._size}, nnz={len(self._data)})"


class LinearSystem:
    """Hold the assembled matrix ``A`` and right-hand-side vector ``b``.

    This container represents the system Ax = b that a future CFD solver
    will solve. No solving is performed here.

    Parameters
    ----------
    matrix : SparseMatrix
        Square coefficient matrix of size ``n``.
    rhs : numpy.ndarray
        Right-hand-side vector of length ``n``.

    Raises
    ------
    TypeError
        If ``matrix`` is not a ``SparseMatrix`` or ``rhs`` is not array-like.
    ValueError
        If the length of ``rhs`` does not match ``matrix.size``.

    Examples
    --------
    >>> mat = SparseMatrix(size=3)
    >>> mat.set(0, 0, 2.0)
    >>> b = np.array([1.0, 0.0, 0.0])
    >>> system = LinearSystem(mat, b)
    >>> print(system.summary())
    LinearSystem: 3x3 matrix, 3 RHS entries, 1 non-zero coefficient(s)
    """

    def __init__(self, matrix: SparseMatrix, rhs: np.ndarray) -> None:
        if not isinstance(matrix, SparseMatrix):
            raise TypeError(f"matrix must be a SparseMatrix, got {type(matrix).__name__}.")

        rhs_array = np.asarray(rhs, dtype=float)
        if rhs_array.ndim != 1:
            raise ValueError("rhs must be a 1D array.")
        if rhs_array.shape[0] != matrix.size:
            raise ValueError(
                f"rhs length ({rhs_array.shape[0]}) must match matrix size ({matrix.size})."
            )

        self.matrix = matrix
        self.rhs = rhs_array

    @property
    def size(self) -> int:
        """Number of unknowns in the system."""
        return self.matrix.size

    def summary(self) -> str:
        """Return a one-line human-readable description of the system.

        Returns
        -------
        str
            Summary string showing dimensions and number of stored coefficients.
        """
        nnz = len(self.matrix._data)
        return (
            f"LinearSystem: {self.size}x{self.size} matrix, "
            f"{self.rhs.shape[0]} RHS entries, "
            f"{nnz} non-zero coefficient(s)"
        )

    def __repr__(self) -> str:
        return f"LinearSystem(size={self.size}, nnz={len(self.matrix._data)})"
