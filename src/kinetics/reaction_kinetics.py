"""
Reaction kinetics calculations for chemical engineering.

This module provides functions to model concentration decay, reaction rates,
and reactor design parameters based on fundamental kinetic theory.
All inputs and outputs use SI units.
"""

import numpy as np


def zero_order_concentration(C0, k, t):
    """
    Calculate concentration for zero-order reaction.

    For a zero-order reaction, concentration decays linearly with time:
    [A]_t = [A]_0 - k*t

    Parameters
    ----------
    C0 : float
        Initial concentration of reactant (mol/L).
        Must be positive.
    k : float
        Zero-order rate constant (mol/(L·s)).
        Must be non-negative.
    t : float
        Time elapsed (s).
        Must be non-negative.

    Returns
    -------
    float
        Concentration at time t (mol/L).
        Cannot be negative (reaction stops at zero concentration).

    Raises
    ------
    TypeError
        If any argument is not numeric (int, float).
    ValueError
        If C0 or k is negative, or t is negative, or if time exceeds
        the reaction completion time (concentration would be negative).

    Examples
    --------
    >>> # Enzyme-catalyzed reaction at saturation
    >>> zero_order_concentration(C0=0.5, k=0.01, t=10)
    0.4

    >>> # Reaction at completion
    >>> zero_order_concentration(C0=1.0, k=0.1, t=10)
    0.0
    """
    # Input validation
    for name, value in [("C0", C0), ("k", k), ("t", t)]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")

    if C0 < 0:
        raise ValueError(f"C0 must be non-negative, got {C0}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")

    # Calculate concentration
    concentration = C0 - k * t

    # Check if reaction is complete
    if concentration < 0:
        raise ValueError(
            f"Time {t} s exceeds reaction completion time {C0/k:.4f} s "
            f"(would result in negative concentration {concentration:.4f} mol/L)"
        )

    return float(concentration)


def first_order_concentration(C0, k, t):
    """
    Calculate concentration for first-order reaction.

    For a first-order reaction, concentration decays exponentially:
    [A]_t = [A]_0 * exp(-k*t)

    Parameters
    ----------
    C0 : float
        Initial concentration of reactant (mol/L).
        Must be positive.
    k : float
        First-order rate constant (s⁻¹).
        Must be non-negative.
    t : float
        Time elapsed (s).
        Must be non-negative.

    Returns
    -------
    float
        Concentration at time t (mol/L).
        Always positive; asymptotically approaches zero.

    Raises
    ------
    TypeError
        If any argument is not numeric (int, float).
    ValueError
        If C0 or k is negative, or t is negative.

    Examples
    --------
    >>> # Drug metabolism in body
    >>> first_order_concentration(C0=100, k=0.1, t=10)  # doctest: +SKIP
    36.787...

    >>> # Radioactive decay
    >>> first_order_concentration(C0=1.0, k=0.693, t=1)  # doctest: +SKIP
    0.5
    """
    # Input validation
    for name, value in [("C0", C0), ("k", k), ("t", t)]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")

    if C0 < 0:
        raise ValueError(f"C0 must be non-negative, got {C0}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")

    concentration = C0 * np.exp(-k * t)
    return float(concentration)


def second_order_concentration(C0, k, t):
    """
    Calculate concentration for second-order reaction.

    For a second-order reaction, inverse concentration increases linearly:
    1/[A]_t = 1/[A]_0 + k*t

    Parameters
    ----------
    C0 : float
        Initial concentration of reactant (mol/L).
        Must be positive (non-zero).
    k : float
        Second-order rate constant (L/(mol·s)).
        Must be non-negative.
    t : float
        Time elapsed (s).
        Must be non-negative.

    Returns
    -------
    float
        Concentration at time t (mol/L).
        Always positive; decays slower than first-order initially.

    Raises
    ------
    TypeError
        If any argument is not numeric (int, float).
    ValueError
        If C0 is not positive, k is negative, t is negative, or
        time parameter (k*t) makes inverse concentration undefined.

    Examples
    --------
    >>> # Bimolecular reaction (equal initial concentrations)
    >>> second_order_concentration(C0=1.0, k=0.1, t=5)
    0.666...

    >>> # Recombination reaction at equilibrium
    >>> second_order_concentration(C0=0.5, k=0.01, t=100)  # doctest: +SKIP
    0.333...
    """
    # Input validation
    for name, value in [("C0", C0), ("k", k), ("t", t)]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")

    if C0 <= 0:
        raise ValueError(f"C0 must be positive, got {C0}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")

    # Calculate inverse concentration: 1/[A]_t = 1/[A]_0 + k*t
    inv_concentration = 1.0 / C0 + k * t

    # Ensure valid result
    if inv_concentration <= 0:
        raise ValueError(
            f"Invalid reaction parameters: 1/[A]_t = {inv_concentration:.4f} "
            f"(must be positive)"
        )

    concentration = 1.0 / inv_concentration
    return float(concentration)


def half_life_first_order(k):
    """
    Calculate half-life for first-order reaction.

    The half-life is the time required for concentration to reach 50% of
    initial value. For first-order reactions, it is independent of C0:
    t_1/2 = ln(2) / k ≈ 0.693 / k

    Parameters
    ----------
    k : float
        First-order rate constant (s⁻¹).
        Must be positive.

    Returns
    -------
    float
        Half-life (s).
        Always positive.

    Raises
    ------
    TypeError
        If k is not numeric (int, float).
    ValueError
        If k is not positive.

    Notes
    -----
    Half-life is independent of initial concentration (a key property of
    first-order reactions). For second-order reactions, half-life increases
    with initial concentration.

    Examples
    --------
    >>> # Drug with first-order metabolism (k=0.1 hr⁻¹ = 2.78e-5 s⁻¹)
    >>> half_life_first_order(k=2.78e-5)  # doctest: +SKIP
    24933.9...

    >>> # Radioactive decay (Carbon-14, k=1.21e-4 yr⁻¹)
    >>> half_life_first_order(k=1.21e-4)  # doctest: +SKIP
    5730.0
    """
    # Input validation
    if not isinstance(k, (int, float)) or isinstance(k, bool):
        raise TypeError(f"k must be numeric (int or float), got {type(k).__name__}")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    half_life = np.log(2) / k
    return float(half_life)


def arrhenius_rate_constant(A, Ea, T):
    """
    Calculate temperature-dependent rate constant from Arrhenius equation.

    The Arrhenius equation describes how reaction rate increases with
    temperature:
    k = A * exp(-Ea / (R*T))

    where R = 8.314 J/(mol·K) is the universal gas constant.

    Parameters
    ----------
    A : float
        Pre-exponential factor (same units as k).
        Must be positive. Represents collision frequency and orientation.
    Ea : float
        Activation energy (J/mol).
        Must be non-negative. Typical range: 50-300 kJ/mol.
    T : float
        Absolute temperature (K).
        Must be positive. Typical range: 250-500 K.

    Returns
    -------
    float
        Rate constant k (same units as A).
        Always positive.

    Raises
    ------
    TypeError
        If any argument is not numeric (int, float).
    ValueError
        If A or T is not positive, or Ea is negative.

    Notes
    -----
    The universal gas constant R = 8.314 J/(mol·K) is hardcoded.
    A 10 K increase in temperature typically increases k by a factor of 2-3
    (depending on Ea).

    Examples
    --------
    >>> # Enzyme catalysis at 25°C (298.15 K)
    >>> # With A=1e8 s⁻¹ and Ea=50 kJ/mol
    >>> k_298 = arrhenius_rate_constant(A=1e8, Ea=50000, T=298.15)

    >>> # Same reaction at 35°C (308.15 K)
    >>> k_308 = arrhenius_rate_constant(A=1e8, Ea=50000, T=308.15)
    >>> ratio = k_308 / k_298  # doctest: +SKIP
    >>> ratio  # ~2.5× faster (typical for Ea=50 kJ/mol)  # doctest: +SKIP
    2.5
    """
    # Input validation
    for name, value in [("A", A), ("Ea", Ea), ("T", T)]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")

    if A <= 0:
        raise ValueError(f"A must be positive, got {A}")
    if Ea < 0:
        raise ValueError(f"Ea must be non-negative, got {Ea}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    # Universal gas constant (J/(mol·K))
    R = 8.314

    # Calculate rate constant
    rate_constant = A * np.exp(-Ea / (R * T))
    return float(rate_constant)


def conversion(initial_concentration, current_concentration):
    """
    Calculate fractional conversion of reactant.

    Conversion is the fraction of reactant consumed:
    X = 1 - [A]_t / [A]_0

    Parameters
    ----------
    initial_concentration : float
        Initial concentration of reactant (mol/L).
        Must be positive.
    current_concentration : float
        Current concentration of reactant (mol/L).
        Must be non-negative and ≤ initial_concentration.

    Returns
    -------
    float
        Fractional conversion (unitless, range 0 to 1).
        0 = no reaction; 1 = complete reaction.

    Raises
    ------
    TypeError
        If either argument is not numeric (int, float).
    ValueError
        If initial_concentration is not positive, or current_concentration
        is negative, or current_concentration exceeds initial_concentration.

    Examples
    --------
    >>> # 50% of reactant consumed
    >>> conversion(initial_concentration=1.0, current_concentration=0.5)
    0.5

    >>> # Complete reaction
    >>> conversion(initial_concentration=2.0, current_concentration=0.0)
    1.0

    >>> # No reaction
    >>> conversion(initial_concentration=1.5, current_concentration=1.5)
    0.0
    """
    # Input validation
    for name, value in [("initial_concentration", initial_concentration),
                        ("current_concentration", current_concentration)]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")

    if initial_concentration <= 0:
        raise ValueError(f"initial_concentration must be positive, got {initial_concentration}")
    if current_concentration < 0:
        raise ValueError(f"current_concentration must be non-negative, got {current_concentration}")
    if current_concentration > initial_concentration:
        raise ValueError(
            f"current_concentration ({current_concentration}) exceeds "
            f"initial_concentration ({initial_concentration})"
        )

    X = 1 - current_concentration / initial_concentration
    return float(X)


def residence_time(volume, volumetric_flow_rate):
    """
    Calculate residence time (mean time in reactor).

    Residence time is the average time a fluid element spends in the reactor:
    τ = V / V̇

    In a continuous stirred-tank reactor (CSTR), this determines conversion
    for a given reaction kinetics.

    Parameters
    ----------
    volume : float
        Reactor volume (L).
        Must be positive.
    volumetric_flow_rate : float
        Volumetric flow rate through reactor (L/s).
        Must be positive.

    Returns
    -------
    float
        Residence time (s).
        Always positive.

    Raises
    ------
    TypeError
        If either argument is not numeric (int, float).
    ValueError
        If either argument is not positive.

    Examples
    --------
    >>> # 100 L reactor with 10 L/s flow
    >>> residence_time(volume=100, volumetric_flow_rate=10)
    10.0

    >>> # 500 mL lab reactor with 5 mL/s flow (after unit conversion)
    >>> residence_time(volume=0.5, volumetric_flow_rate=0.005)
    100.0
    """
    # Input validation
    for name, value in [("volume", volume), ("volumetric_flow_rate", volumetric_flow_rate)]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")

    if volume <= 0:
        raise ValueError(f"volume must be positive, got {volume}")
    if volumetric_flow_rate <= 0:
        raise ValueError(f"volumetric_flow_rate must be positive, got {volumetric_flow_rate}")

    tau = volume / volumetric_flow_rate
    return float(tau)
