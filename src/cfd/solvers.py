"""Iterative linear solvers for the CFD linear systems assembled in earlier sprints.

Both solvers here are classic textbook methods for solving Ax = b:

- Jacobi:       every unknown is updated from the *old* values of its
                neighbours, all at once.
- Gauss-Seidel: every unknown is updated from whatever the *latest* values
                of its neighbours happen to be, one at a time. Because it
                uses fresher information, Gauss-Seidel usually converges
                faster than Jacobi for the same problem.

Both are "relaxation" methods: they start from a guess and repeatedly nudge
it closer to the true solution until the change between iterations
(the residual) drops below a chosen tolerance.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .linear_system import LinearSystem


class _IterativeSolver:
    """Shared setup for the small iterative solvers below."""

    def __init__(self, max_iterations: int = 2000, tolerance: float = 1e-8) -> None:
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an integer.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise TypeError("tolerance must be a number.")
        if tolerance <= 0:
            raise ValueError("tolerance must be a positive number.")

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Filled in by solve(): how many iterations actually ran, whether the
        # residual dropped below tolerance, and the residual at every step
        # (handy for plotting a convergence curve while learning the method).
        self.iterations_run = 0
        self.converged = False
        self.residual_history: List[float] = []

    @staticmethod
    def _validate_system(system: LinearSystem) -> None:
        if not isinstance(system, LinearSystem):
            raise TypeError(f"system must be a LinearSystem, got {type(system).__name__}.")

    @staticmethod
    def _check_diagonal(diagonal: np.ndarray) -> None:
        # Both methods divide by the diagonal entry of each row, so a zero
        # there would mean dividing by zero.
        if np.any(np.isclose(diagonal, 0.0)):
            raise ValueError("matrix has a zero diagonal entry; iteration cannot proceed.")

    def _initial_guess(self, n: int, initial_guess: Optional[np.ndarray]) -> np.ndarray:
        if initial_guess is None:
            return np.zeros(n, dtype=float)
        guess = np.asarray(initial_guess, dtype=float)
        if guess.shape != (n,):
            raise ValueError(f"initial_guess must have shape ({n},).")
        return guess.copy()

    def solve(self, system: LinearSystem, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement solve().")


class JacobiSolver(_IterativeSolver):
    """Solve Ax = b with the Jacobi method.

    Every entry of x is updated using only the values of x from the
    *previous* iteration, so the update order does not matter.
    """

    def solve(self, system: LinearSystem, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        self._validate_system(system)
        A = system.matrix.to_array()
        b = system.rhs
        n = system.size
        diagonal = np.diag(A)
        self._check_diagonal(diagonal)

        x = self._initial_guess(n, initial_guess)
        self.residual_history = []
        self.converged = False

        for iteration in range(self.max_iterations):
            # Everything on the row except the diagonal term, using last
            # iteration's x throughout — this is what makes it "Jacobi".
            off_diagonal_sum = A @ x - diagonal * x
            x_new = (b - off_diagonal_sum) / diagonal

            residual = float(np.linalg.norm(x_new - x))
            self.residual_history.append(residual)
            x = x_new
            self.iterations_run = iteration + 1

            if residual < self.tolerance:
                self.converged = True
                break

        return x


class GaussSeidelSolver(_IterativeSolver):
    """Solve Ax = b with the Gauss-Seidel method.

    Unlike Jacobi, each entry of x is updated in place, so later entries in
    the same sweep already see the newest values of earlier entries.
    """

    def solve(self, system: LinearSystem, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        self._validate_system(system)
        A = system.matrix.to_array()
        b = system.rhs
        n = system.size
        diagonal = np.diag(A)
        self._check_diagonal(diagonal)

        x = self._initial_guess(n, initial_guess)
        self.residual_history = []
        self.converged = False

        for iteration in range(self.max_iterations):
            x_before_sweep = x.copy()
            for i in range(n):
                # A[i] @ x already reflects this sweep's updates for indices
                # before i, and last sweep's values for indices after i.
                row_sum = A[i] @ x - A[i, i] * x[i]
                x[i] = (b[i] - row_sum) / A[i, i]

            residual = float(np.linalg.norm(x - x_before_sweep))
            self.residual_history.append(residual)
            self.iterations_run = iteration + 1

            if residual < self.tolerance:
                self.converged = True
                break

        return x
