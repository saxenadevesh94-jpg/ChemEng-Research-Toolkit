"""Lightweight equation representation for CFD solvers."""

from __future__ import annotations

from typing import List


class Equation:
    """Represent a single CFD equation as lhs = rhs + source terms.

    This class does not solve anything; it stores the symbolic structure
    of an equation so that a future solver can consume it.

    Parameters
    ----------
    lhs : str
        String describing the left-hand side operator or term (e.g. ``"laplacian(U)"``).
    rhs : str
        String describing the right-hand side (e.g. ``"0"`` or ``"ddt(U)"``).

    Raises
    ------
    TypeError
        If ``lhs`` or ``rhs`` are not strings.
    ValueError
        If ``lhs`` or ``rhs`` are empty or whitespace-only.

    Examples
    --------
    >>> eq = Equation(lhs="laplacian(U)", rhs="0")
    >>> eq.add_source("pressure_gradient")
    >>> print(eq)
    laplacian(U) = 0 + pressure_gradient
    """

    def __init__(self, lhs: str, rhs: str) -> None:
        self.lhs = self._validate_term(lhs, "lhs")
        self.rhs = self._validate_term(rhs, "rhs")
        # Source terms accumulate here; solvers will sum them with the rhs.
        self._sources: List[str] = []

    @staticmethod
    def _validate_term(value: object, name: str) -> str:
        """Check that a term is a non-empty string."""
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty or whitespace-only.")
        return stripped

    def add_source(self, term: str) -> None:
        """Append a source term to the right-hand side.

        Parameters
        ----------
        term : str
            Symbolic name of the source (e.g. ``"pressure_gradient"``).

        Raises
        ------
        TypeError
            If ``term`` is not a string.
        ValueError
            If ``term`` is empty or whitespace-only.
        """
        self._sources.append(self._validate_term(term, "term"))

    @property
    def sources(self) -> List[str]:
        """Return a copy of the current source term list."""
        return list(self._sources)

    def __str__(self) -> str:
        """Return a human-readable representation of the equation."""
        parts = [self.rhs] + self._sources
        rhs_full = " + ".join(parts)
        return f"{self.lhs} = {rhs_full}"

    def __repr__(self) -> str:
        return f"Equation(lhs={self.lhs!r}, rhs={self.rhs!r}, sources={self._sources!r})"
