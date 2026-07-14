import pytest

from src.cfd import Equation


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_equation_stores_lhs_and_rhs():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    assert eq.lhs == "laplacian(U)"
    assert eq.rhs == "0"


def test_equation_trims_surrounding_whitespace():
    eq = Equation(lhs="  ddt(T)  ", rhs="  source  ")
    assert eq.lhs == "ddt(T)"
    assert eq.rhs == "source"


def test_equation_starts_with_no_sources():
    eq = Equation(lhs="laplacian(p)", rhs="0")
    assert eq.sources == []


# ---------------------------------------------------------------------------
# Adding source terms
# ---------------------------------------------------------------------------

def test_add_single_source():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    eq.add_source("pressure_gradient")
    assert eq.sources == ["pressure_gradient"]


def test_add_multiple_sources():
    eq = Equation(lhs="ddt(U)", rhs="0")
    eq.add_source("pressure_gradient")
    eq.add_source("viscous_diffusion")
    eq.add_source("gravity")
    assert eq.sources == ["pressure_gradient", "viscous_diffusion", "gravity"]


def test_sources_property_returns_copy():
    """Mutating the returned list must not change the internal state."""
    eq = Equation(lhs="laplacian(T)", rhs="0")
    eq.add_source("heat_source")
    snapshot = eq.sources
    snapshot.append("injected")
    assert eq.sources == ["heat_source"]


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

def test_str_no_sources():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    assert str(eq) == "laplacian(U) = 0"


def test_str_with_one_source():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    eq.add_source("pressure_gradient")
    assert str(eq) == "laplacian(U) = 0 + pressure_gradient"


def test_str_with_multiple_sources():
    eq = Equation(lhs="ddt(U)", rhs="0")
    eq.add_source("pressure_gradient")
    eq.add_source("viscous_diffusion")
    assert str(eq) == "ddt(U) = 0 + pressure_gradient + viscous_diffusion"


def test_repr_contains_key_fields():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    r = repr(eq)
    assert "laplacian(U)" in r
    assert "0" in r


# ---------------------------------------------------------------------------
# Validation – invalid inputs
# ---------------------------------------------------------------------------

def test_lhs_must_be_string():
    with pytest.raises(TypeError):
        Equation(lhs=42, rhs="0")


def test_rhs_must_be_string():
    with pytest.raises(TypeError):
        Equation(lhs="laplacian(U)", rhs=None)


def test_lhs_must_not_be_empty():
    with pytest.raises(ValueError):
        Equation(lhs="", rhs="0")


def test_rhs_must_not_be_empty():
    with pytest.raises(ValueError):
        Equation(lhs="laplacian(U)", rhs="   ")


def test_source_must_be_string():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    with pytest.raises(TypeError):
        eq.add_source(123)


def test_source_must_not_be_empty():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    with pytest.raises(ValueError):
        eq.add_source("")


def test_source_must_not_be_whitespace_only():
    eq = Equation(lhs="laplacian(U)", rhs="0")
    with pytest.raises(ValueError):
        eq.add_source("   ")


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_equation_is_independent_between_instances():
    """Sources added to one equation must not appear in another."""
    eq1 = Equation(lhs="laplacian(U)", rhs="0")
    eq2 = Equation(lhs="laplacian(T)", rhs="0")
    eq1.add_source("pressure_gradient")
    assert eq2.sources == []


def test_str_output_matches_example_from_spec():
    """Reproduce the exact example given in the sprint specification."""
    eq = Equation(lhs="laplacian(U)", rhs="0")
    eq.add_source("pressure_gradient")
    assert str(eq) == "laplacian(U) = 0 + pressure_gradient"


def test_add_source_returns_none():
    """add_source is a void operation; it must not return a value."""
    eq = Equation(lhs="laplacian(U)", rhs="0")
    result = eq.add_source("buoyancy")
    assert result is None
