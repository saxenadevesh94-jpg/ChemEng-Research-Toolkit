import math


def reynolds_number(density, velocity, diameter, viscosity):
    """
    Calculate the Reynolds Number.

    The Reynolds Number compares inertial forces to viscous forces.
    Re = ρ × v × D / μ

    Parameters
    ----------
    density : float
        Fluid density in kg/m³.
    velocity : float
        Flow velocity in m/s.
    diameter : float
        Characteristic length (pipe diameter, droplet diameter) in m.
    viscosity : float
        Dynamic viscosity in Pa·s (kg/m·s).

    Returns
    -------
    float
        The Reynolds Number (dimensionless).

    Raises
    ------
    ValueError
        If any value is negative or if required values are zero.
    TypeError
        If inputs are not numeric.
    """
    if not isinstance(density, (int, float)) or not isinstance(velocity, (int, float)) \
            or not isinstance(diameter, (int, float)) or not isinstance(viscosity, (int, float)):
        raise TypeError("All inputs must be numeric values.")

    if density < 0:
        raise ValueError("Density cannot be negative.")
    if velocity < 0:
        raise ValueError("Velocity cannot be negative.")
    if diameter < 0:
        raise ValueError("Diameter cannot be negative.")
    if viscosity < 0:
        raise ValueError("Viscosity cannot be negative.")

    if viscosity == 0:
        raise ValueError("Viscosity cannot be zero.")

    re = (density * velocity * diameter) / viscosity

    return re


def weber_number(density, velocity, diameter, surface_tension):
    """
    Calculate the Weber Number.

    The Weber Number compares inertial forces to surface tension forces.
    We = ρ × v² × D / σ

    Parameters
    ----------
    density : float
        Fluid density in kg/m³.
    velocity : float
        Flow velocity in m/s.
    diameter : float
        Characteristic length (droplet diameter, jet diameter) in m.
    surface_tension : float
        Surface tension in N/m (kg/s²).

    Returns
    -------
    float
        The Weber Number (dimensionless).

    Raises
    ------
    ValueError
        If any value is negative or if required values are zero.
    TypeError
        If inputs are not numeric.
    """
    if not isinstance(density, (int, float)) or not isinstance(velocity, (int, float)) \
            or not isinstance(diameter, (int, float)) or not isinstance(surface_tension, (int, float)):
        raise TypeError("All inputs must be numeric values.")

    if density < 0:
        raise ValueError("Density cannot be negative.")
    if velocity < 0:
        raise ValueError("Velocity cannot be negative.")
    if diameter < 0:
        raise ValueError("Diameter cannot be negative.")
    if surface_tension < 0:
        raise ValueError("Surface tension cannot be negative.")

    if surface_tension == 0:
        raise ValueError("Surface tension cannot be zero.")

    we = (density * velocity ** 2 * diameter) / surface_tension

    return we


def capillary_number(viscosity, velocity, surface_tension):
    """
    Calculate the Capillary Number.

    The Capillary Number compares viscous forces to surface tension forces.
    Ca = μ × v / σ

    Parameters
    ----------
    viscosity : float
        Dynamic viscosity in Pa·s (kg/m·s).
    velocity : float
        Flow velocity in m/s.
    surface_tension : float
        Surface tension in N/m (kg/s²).

    Returns
    -------
    float
        The Capillary Number (dimensionless).

    Raises
    ------
    ValueError
        If any value is negative or if required values are zero.
    TypeError
        If inputs are not numeric.
    """
    if not isinstance(viscosity, (int, float)) or not isinstance(velocity, (int, float)) \
            or not isinstance(surface_tension, (int, float)):
        raise TypeError("All inputs must be numeric values.")

    if viscosity < 0:
        raise ValueError("Viscosity cannot be negative.")
    if velocity < 0:
        raise ValueError("Velocity cannot be negative.")
    if surface_tension < 0:
        raise ValueError("Surface tension cannot be negative.")

    if surface_tension == 0:
        raise ValueError("Surface tension cannot be zero.")

    ca = (viscosity * velocity) / surface_tension

    return ca


def froude_number(velocity, diameter, gravity=9.81):
    """
    Calculate the Froude Number.

    The Froude Number compares inertial forces to gravitational forces.
    Fr = v / √(g × D)

    Parameters
    ----------
    velocity : float
        Flow velocity in m/s.
    diameter : float
        Characteristic length (pipe diameter, droplet diameter) in m.
    gravity : float, optional
        Acceleration due to gravity in m/s². Default is 9.81.

    Returns
    -------
    float
        The Froude Number (dimensionless).

    Raises
    ------
    ValueError
        If any value is negative or if required values are zero.
    TypeError
        If inputs are not numeric.
    """
    if not isinstance(velocity, (int, float)) or not isinstance(diameter, (int, float)) \
            or not isinstance(gravity, (int, float)):
        raise TypeError("All inputs must be numeric values.")

    if velocity < 0:
        raise ValueError("Velocity cannot be negative.")
    if diameter < 0:
        raise ValueError("Diameter cannot be negative.")
    if gravity < 0:
        raise ValueError("Gravity cannot be negative.")

    if diameter == 0:
        raise ValueError("Diameter cannot be zero.")

    fr = velocity / math.sqrt(gravity * diameter)

    return fr


def bond_number(density, diameter, surface_tension, gravity=9.81):
    """
    Calculate the Bond Number (Eötvös Number).

    The Bond Number compares gravitational forces to surface tension forces.
    Bo = ρ × g × D² / σ

    Parameters
    ----------
    density : float
        Fluid density in kg/m³.
    diameter : float
        Characteristic length (droplet diameter, bubble diameter) in m.
    surface_tension : float
        Surface tension in N/m (kg/s²).
    gravity : float, optional
        Acceleration due to gravity in m/s². Default is 9.81.

    Returns
    -------
    float
        The Bond Number (dimensionless).

    Raises
    ------
    ValueError
        If any value is negative or if required values are zero.
    TypeError
        If inputs are not numeric.
    """
    if not isinstance(density, (int, float)) or not isinstance(diameter, (int, float)) \
            or not isinstance(surface_tension, (int, float)) or not isinstance(gravity, (int, float)):
        raise TypeError("All inputs must be numeric values.")

    if density < 0:
        raise ValueError("Density cannot be negative.")
    if diameter < 0:
        raise ValueError("Diameter cannot be negative.")
    if surface_tension < 0:
        raise ValueError("Surface tension cannot be negative.")
    if gravity < 0:
        raise ValueError("Gravity cannot be negative.")

    if surface_tension == 0:
        raise ValueError("Surface tension cannot be zero.")

    bo = (density * gravity * diameter ** 2) / surface_tension

    return bo


def ohnesorge_number(viscosity, density, diameter, surface_tension):
    """
    Calculate the Ohnesorge Number.

    The Ohnesorge Number combines viscous, inertial, and surface tension effects.
    Oh = μ / √(ρ × σ × D)

    Parameters
    ----------
    viscosity : float
        Dynamic viscosity in Pa·s (kg/m·s).
    density : float
        Fluid density in kg/m³.
    diameter : float
        Characteristic length (droplet diameter) in m.
    surface_tension : float
        Surface tension in N/m (kg/s²).

    Returns
    -------
    float
        The Ohnesorge Number (dimensionless).

    Raises
    ------
    ValueError
        If any value is negative or if required values are zero.
    TypeError
        If inputs are not numeric.
    """
    if not isinstance(viscosity, (int, float)) or not isinstance(density, (int, float)) \
            or not isinstance(diameter, (int, float)) or not isinstance(surface_tension, (int, float)):
        raise TypeError("All inputs must be numeric values.")

    if viscosity < 0:
        raise ValueError("Viscosity cannot be negative.")
    if density < 0:
        raise ValueError("Density cannot be negative.")
    if diameter < 0:
        raise ValueError("Diameter cannot be negative.")
    if surface_tension < 0:
        raise ValueError("Surface tension cannot be negative.")

    if density == 0 or surface_tension == 0 or diameter == 0:
        raise ValueError("Density, surface tension, and diameter cannot be zero.")

    oh = viscosity / math.sqrt(density * surface_tension * diameter)

    return oh
