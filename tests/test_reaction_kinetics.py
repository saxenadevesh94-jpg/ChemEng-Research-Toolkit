"""
Comprehensive test suite for reaction kinetics module.

Tests cover realistic chemical engineering scenarios including:
- Reaction order comparisons
- Temperature effects (Arrhenius equation)
- Half-life predictions
- Conversion metrics for reactor design
- Residence time calculations
- Error handling and edge cases
"""

import pytest
import numpy as np
from src.kinetics import (
    zero_order_concentration,
    first_order_concentration,
    second_order_concentration,
    half_life_first_order,
    arrhenius_rate_constant,
    conversion,
    residence_time,
)


# ============================================================================
# Zero-Order Reaction Tests
# ============================================================================

class TestZeroOrderConcentration:
    """Tests for zero-order concentration decay [A]_t = [A]_0 - k*t"""

    def test_zero_order_no_reaction(self):
        """No reaction has occurred (t=0)"""
        result = zero_order_concentration(C0=1.0, k=0.1, t=0)
        assert result == pytest.approx(1.0)

    def test_zero_order_partial_reaction(self):
        """Partial reaction: 50% conversion after time t"""
        # [A]_0 = 1.0 mol/L, k = 0.1 mol/(L·s), at t=5s: [A]_t = 1.0 - 0.1*5 = 0.5
        result = zero_order_concentration(C0=1.0, k=0.1, t=5)
        assert result == pytest.approx(0.5)

    def test_zero_order_complete_reaction(self):
        """Reaction goes to completion (all reactant consumed)"""
        # At t = C0/k, concentration reaches zero
        result = zero_order_concentration(C0=2.0, k=0.4, t=5)
        assert result == pytest.approx(0.0)

    def test_zero_order_enzyme_kinetics(self):
        """
        Real scenario: Enzyme-catalyzed reaction at saturation.
        Michaelis-Menten enzyme shows zero-order kinetics when [S] >> Km.
        Example: Enzymatic hydrolysis of substrate at high concentration.
        """
        C0 = 0.5  # mol/L (high substrate concentration)
        k = 0.02  # mol/(L·s) (enzyme-limited rate)
        t = 10  # seconds
        result = zero_order_concentration(C0, k, t)
        assert result == pytest.approx(0.3)

    def test_zero_order_exceeds_completion(self):
        """Error: time parameter exceeds reaction completion"""
        with pytest.raises(ValueError, match="exceeds reaction completion time"):
            zero_order_concentration(C0=1.0, k=0.1, t=15)

    def test_zero_order_negative_C0(self):
        """Error: negative initial concentration"""
        with pytest.raises(ValueError, match="C0 must be non-negative"):
            zero_order_concentration(C0=-0.5, k=0.1, t=5)

    def test_zero_order_negative_k(self):
        """Error: negative rate constant (physically impossible)"""
        with pytest.raises(ValueError, match="k must be non-negative"):
            zero_order_concentration(C0=1.0, k=-0.05, t=5)

    def test_zero_order_negative_t(self):
        """Error: negative time (physically impossible)"""
        with pytest.raises(ValueError, match="t must be non-negative"):
            zero_order_concentration(C0=1.0, k=0.1, t=-2)

    def test_zero_order_non_numeric_C0(self):
        """Error: non-numeric initial concentration"""
        with pytest.raises(TypeError, match="C0 must be numeric"):
            zero_order_concentration(C0="1.0", k=0.1, t=5)

    def test_zero_order_boolean_input(self):
        """Error: boolean values (which are technically numeric in Python)"""
        with pytest.raises(TypeError, match="must be numeric"):
            zero_order_concentration(C0=True, k=0.1, t=5)


# ============================================================================
# First-Order Reaction Tests
# ============================================================================

class TestFirstOrderConcentration:
    """Tests for first-order concentration decay [A]_t = [A]_0 * exp(-k*t)"""

    def test_first_order_no_reaction(self):
        """No reaction has occurred (t=0)"""
        result = first_order_concentration(C0=1.0, k=0.1, t=0)
        assert result == pytest.approx(1.0)

    def test_first_order_half_life(self):
        """At t=t_1/2, concentration reaches 50% of initial"""
        # For first-order: t_1/2 = ln(2)/k = 0.693/0.1 = 6.93 s
        k = 0.1  # s⁻¹
        t_half = np.log(2) / k
        result = first_order_concentration(C0=2.0, k=k, t=t_half)
        assert result == pytest.approx(1.0, rel=1e-5)  # Should be exactly 0.5*2.0

    def test_first_order_partial_reaction(self):
        """Partial reaction: calculate remaining concentration"""
        # [A]_0 = 5 mol/L, k = 0.05 s⁻¹, t = 20 s
        # [A]_t = 5 * exp(-0.05*20) = 5 * exp(-1) = 5 * 0.3679 = 1.839
        result = first_order_concentration(C0=5.0, k=0.05, t=20)
        assert result == pytest.approx(1.839, rel=1e-3)

    def test_first_order_drug_metabolism(self):
        """
        Real scenario: Drug metabolism in human body.
        Elimination rate constant for acetaminophen: ~0.0003 s⁻¹
        Plasma concentration after 1 hour (3600 s).
        """
        C0 = 250  # mg/L (typical peak plasma concentration)
        k = 0.0003  # s⁻¹ (first-order elimination rate)
        t = 3600  # 1 hour in seconds
        result = first_order_concentration(C0, k, t)
        expected = 250 * np.exp(-0.0003 * 3600)
        assert result == pytest.approx(expected)

    def test_first_order_radioactive_decay(self):
        """
        Real scenario: Carbon-14 radioactive decay.
        Half-life of C-14 = 5730 years.
        Rate constant k = ln(2) / t_1/2 = 1.21e-4 yr⁻¹
        After 11,460 years (2 half-lives), concentration ≈ 25% of initial.
        """
        # Using seconds for consistency (11,460 years ≈ 3.62e11 s)
        # and k ≈ 1.91e-12 s⁻¹
        t_half_sec = 5730 * 365.25 * 24 * 3600  # Convert years to seconds
        k = np.log(2) / t_half_sec
        C0 = 100  # arbitrary initial amount
        t = 2 * t_half_sec  # 2 half-lives
        result = first_order_concentration(C0, k, t)
        assert result == pytest.approx(25.0, rel=1e-5)

    def test_first_order_negative_k(self):
        """Error: negative rate constant"""
        with pytest.raises(ValueError, match="k must be non-negative"):
            first_order_concentration(C0=1.0, k=-0.05, t=5)

    def test_first_order_non_numeric_t(self):
        """Error: non-numeric time parameter"""
        with pytest.raises(TypeError, match="t must be numeric"):
            first_order_concentration(C0=1.0, k=0.1, t=[1, 2, 3])


# ============================================================================
# Second-Order Reaction Tests
# ============================================================================

class TestSecondOrderConcentration:
    """Tests for second-order concentration decay 1/[A]_t = 1/[A]_0 + k*t"""

    def test_second_order_no_reaction(self):
        """No reaction has occurred (t=0)"""
        result = second_order_concentration(C0=1.0, k=0.1, t=0)
        assert result == pytest.approx(1.0)

    def test_second_order_partial_reaction(self):
        """Partial reaction: calculate remaining concentration"""
        # 1/[A]_t = 1/1.0 + 0.1*5 = 1 + 0.5 = 1.5
        # [A]_t = 1/1.5 = 0.667
        result = second_order_concentration(C0=1.0, k=0.1, t=5)
        assert result == pytest.approx(1.0 / 1.5, rel=1e-5)

    def test_second_order_slower_decay_than_first(self):
        """
        Second-order reactions decay slower than first-order initially.
        Compare same initial concentration and similar rate parameters at short times.
        """
        C0 = 1.0
        t = 1.0
        # First-order: [A]_t = 1.0 * exp(-0.1*1) ≈ 0.905
        # Second-order: 1/[A]_t = 1 + 0.1*1 = 1.1, [A]_t ≈ 0.909
        c_first = first_order_concentration(C0=1.0, k=0.1, t=1.0)
        c_second = second_order_concentration(C0=1.0, k=0.1, t=1.0)
        assert c_second > c_first  # Less decay for second-order

    def test_second_order_bimolecular_reaction(self):
        """
        Real scenario: Bimolecular reaction (A + B → products).
        When [A]_0 = [B]_0 (equal stoichiometry), follows second-order kinetics.
        Example: Recombination of radicals.
        """
        C0 = 0.5  # mol/L (initial concentration of reactive species)
        k = 0.01  # L/(mol·s) (second-order rate constant)
        t = 100  # seconds
        result = second_order_concentration(C0, k, t)
        expected = 1.0 / (1.0/0.5 + 0.01*100)  # = 1/(2 + 1) = 0.333
        assert result == pytest.approx(expected)

    def test_second_order_dimerization(self):
        """
        Real scenario: Dimerization reaction (2A → A₂).
        Example: Protein oligomerization, polymer chain termination.
        """
        C0 = 0.1  # mol/L
        k = 0.005  # L/(mol·s)
        t = 50  # seconds
        result = second_order_concentration(C0, k, t)
        inv_c = 1.0/C0 + k*t
        assert result == pytest.approx(1.0 / inv_c)

    def test_second_order_zero_initial_concentration(self):
        """Error: zero initial concentration (division by zero)"""
        with pytest.raises(ValueError, match="C0 must be positive"):
            second_order_concentration(C0=0.0, k=0.1, t=5)

    def test_second_order_negative_C0(self):
        """Error: negative initial concentration"""
        with pytest.raises(ValueError, match="C0 must be positive"):
            second_order_concentration(C0=-0.5, k=0.1, t=5)


# ============================================================================
# Half-Life Tests
# ============================================================================

class TestHalfLifeFirstOrder:
    """Tests for half-life of first-order reactions: t_1/2 = ln(2) / k"""

    def test_half_life_basic(self):
        """Basic half-life calculation"""
        k = 0.1  # s⁻¹
        result = half_life_first_order(k)
        expected = np.log(2) / 0.1  # 6.931 s
        assert result == pytest.approx(expected)

    def test_half_life_radioactive_decay(self):
        """
        Half-life of Uranium-235: 7.04e8 years.
        Rate constant k = ln(2) / (7.04e8 * 365.25 * 24 * 3600) s⁻¹
        Verify t_1/2 calculated from k equals original value.
        """
        t_half_original = 7.04e8 * 365.25 * 24 * 3600  # Convert to seconds
        k = np.log(2) / t_half_original
        t_half_calculated = half_life_first_order(k)
        assert t_half_calculated == pytest.approx(t_half_original, rel=1e-5)

    def test_half_life_fast_reaction(self):
        """Fast reaction: large k → short half-life"""
        k = 10.0  # s⁻¹ (very fast)
        result = half_life_first_order(k)
        assert result == pytest.approx(0.0693, rel=1e-3)

    def test_half_life_slow_reaction(self):
        """Slow reaction: small k → long half-life"""
        k = 0.0001  # s⁻¹ (very slow)
        result = half_life_first_order(k)
        expected = np.log(2) / 0.0001
        assert result == pytest.approx(expected)

    def test_half_life_consecutive_application(self):
        """
        After time t_1/2, concentration is 50%.
        After time 2*t_1/2, concentration is 25%.
        Verify with first_order_concentration function.
        """
        C0 = 100
        k = 0.1
        t_half = half_life_first_order(k)
        c_at_half_life = first_order_concentration(C0, k, t_half)
        c_at_two_half_lives = first_order_concentration(C0, k, 2*t_half)
        assert c_at_half_life == pytest.approx(50, rel=1e-5)
        assert c_at_two_half_lives == pytest.approx(25, rel=1e-5)

    def test_half_life_zero_k(self):
        """Error: zero rate constant (infinite half-life)"""
        with pytest.raises(ValueError, match="k must be positive"):
            half_life_first_order(k=0.0)

    def test_half_life_negative_k(self):
        """Error: negative rate constant"""
        with pytest.raises(ValueError, match="k must be positive"):
            half_life_first_order(k=-0.05)

    def test_half_life_non_numeric(self):
        """Error: non-numeric rate constant"""
        with pytest.raises(TypeError, match="k must be numeric"):
            half_life_first_order(k="0.1")


# ============================================================================
# Arrhenius Equation Tests
# ============================================================================

class TestArrheniusRateConstant:
    """Tests for temperature-dependent rate constant: k = A * exp(-Ea / (R*T))"""

    def test_arrhenius_basic(self):
        """Basic Arrhenius calculation"""
        A = 1e8  # s⁻¹
        Ea = 50000  # J/mol (50 kJ/mol)
        T = 298.15  # K (25°C)
        result = arrhenius_rate_constant(A, Ea, T)
        R = 8.314
        expected = A * np.exp(-Ea / (R * T))
        assert result == pytest.approx(expected)

    def test_arrhenius_temperature_effect_modest_ea(self):
        """
        Temperature increase increases rate constant.
        For Ea=50 kJ/mol, increasing temperature from 298.15 K to 308.15 K
        increases the rate constant by approximately 1.92×.
        """
        A = 1e8
        Ea = 50000  # 50 kJ/mol
        k_298 = arrhenius_rate_constant(A, Ea, 298.15)
        k_308 = arrhenius_rate_constant(A, Ea, 308.15)  # +10 K
        ratio = k_308 / k_298
        assert ratio == pytest.approx(1.9243, rel=0.02)

    def test_arrhenius_temperature_effect_high_ea(self):
        """
        High activation energy: 200 kJ/mol.
        10 K increase yields larger rate multiplication (~4-5×).
        """
        A = 1e8
        Ea = 200000  # 200 kJ/mol
        k_300 = arrhenius_rate_constant(A, Ea, 300)
        k_310 = arrhenius_rate_constant(A, Ea, 310)
        ratio = k_310 / k_300
        assert ratio > 4.0  # High Ea → steep temperature dependence

    def test_arrhenius_low_activation_energy(self):
        """
        Low activation energy: reaction is temperature-insensitive.
        Example: Diffusion-limited reactions.
        """
        A = 1e10
        Ea = 5000  # 5 kJ/mol (low)
        k_298 = arrhenius_rate_constant(A, Ea, 298.15)
        k_308 = arrhenius_rate_constant(A, Ea, 308.15)
        ratio = k_308 / k_298
        assert ratio < 1.5  # Small increase

    def test_arrhenius_zero_activation_energy(self):
        """
        Zero activation energy (rare).
        Rate constant equals pre-exponential factor.
        """
        A = 0.5
        Ea = 0
        T = 300
        result = arrhenius_rate_constant(A, Ea, T)
        assert result == pytest.approx(0.5)

    def test_arrhenius_negative_ea(self):
        """Error: negative activation energy (non-physical)"""
        with pytest.raises(ValueError, match="Ea must be non-negative"):
            arrhenius_rate_constant(A=1e8, Ea=-10000, T=298.15)

    def test_arrhenius_zero_A(self):
        """Error: zero pre-exponential factor (reaction impossible)"""
        with pytest.raises(ValueError, match="A must be positive"):
            arrhenius_rate_constant(A=0.0, Ea=50000, T=298.15)

    def test_arrhenius_very_high_temperature(self):
        """Reaction at very high temperature (e.g., combustion)"""
        A = 1e10
        Ea = 100000
        T = 1500  # K

        result = arrhenius_rate_constant(A, Ea, T)

        R = 8.314
        expected = A * np.exp(-Ea / (R * T))

        assert result == pytest.approx(expected)


# ============================================================================
# Conversion Tests
# ============================================================================

class TestConversion:
    """Tests for conversion metric: X = 1 - [A]_t / [A]_0"""

    def test_conversion_zero(self):
        """No reaction: X = 0"""
        result = conversion(initial_concentration=1.0, current_concentration=1.0)
        assert result == pytest.approx(0.0)

    def test_conversion_fifty_percent(self):
        """50% conversion"""
        result = conversion(initial_concentration=2.0, current_concentration=1.0)
        assert result == pytest.approx(0.5)

    def test_conversion_complete(self):
        """Complete reaction: X = 1"""
        result = conversion(initial_concentration=5.0, current_concentration=0.0)
        assert result == pytest.approx(1.0)

    def test_conversion_from_first_order_kinetics(self):
        """
        Calculate conversion from first-order reaction.
        If C0=1 mol/L, after first_order_concentration gives [A]_t,
        conversion X = 1 - [A]_t / C0.
        """
        C0 = 1.0
        k = 0.05
        t = 20  # seconds
        c_t = first_order_concentration(C0, k, t)
        X = conversion(C0, c_t)
        # Verify: X = 1 - exp(-kt)
        expected_X = 1 - np.exp(-k * t)
        assert X == pytest.approx(expected_X)

    def test_conversion_batch_reactor_example(self):
        """
        Real scenario: Batch reactor for hydrolysis.
        Initial API concentration: 2.5 mol/L
        After 30 min (1800 s), concentration drops to 1.875 mol/L (25% conversion).
        """
        X = conversion(initial_concentration=2.5, current_concentration=1.875)
        assert X == pytest.approx(0.25)

    def test_conversion_current_exceeds_initial(self):
        """Error: impossible condition (concentration increased)"""
        with pytest.raises(ValueError, match="current_concentration.*exceeds"):
            conversion(initial_concentration=1.0, current_concentration=1.5)

    def test_conversion_negative_initial(self):
        """Error: negative initial concentration"""
        with pytest.raises(ValueError, match="initial_concentration must be positive"):
            conversion(initial_concentration=-1.0, current_concentration=0.5)

    def test_conversion_zero_initial(self):
        """Error: zero initial concentration (division by zero)"""
        with pytest.raises(ValueError, match="initial_concentration must be positive"):
            conversion(initial_concentration=0.0, current_concentration=0.0)

    def test_conversion_negative_current(self):
        """Error: negative current concentration"""
        with pytest.raises(ValueError, match="current_concentration must be non-negative"):
            conversion(initial_concentration=1.0, current_concentration=-0.1)


# ============================================================================
# Residence Time Tests
# ============================================================================

class TestResidenceTime:
    """Tests for residence time: τ = V / V̇"""

    def test_residence_time_basic(self):
        """Basic residence time calculation"""
        result = residence_time(volume=100, volumetric_flow_rate=10)
        assert result == pytest.approx(10.0)

    def test_residence_time_cstr(self):
        """
        Real scenario: Continuous stirred-tank reactor (CSTR).
        Reactor volume: 500 L
        Flow rate: 50 L/min = 0.833 L/s
        Residence time: 500 / 0.833 ≈ 600 s ≈ 10 min
        """
        volume_L = 500
        flow_rate_L_min = 50
        flow_rate_L_s = flow_rate_L_min / 60
        result = residence_time(volume_L, flow_rate_L_s)
        assert result == pytest.approx(600, rel=1e-2)

    def test_residence_time_microreactor(self):
        """
        Microreactor: very small volume and flow → very short residence time.
        Volume: 1 mL = 0.001 L
        Flow rate: 0.1 mL/s = 0.0001 L/s
        Residence time: 10 seconds
        """
        result = residence_time(volume=0.001, volumetric_flow_rate=0.0001)
        assert result == pytest.approx(10.0)

    def test_residence_time_scaling(self):
        """Doubling volume doubles residence time (same flow)"""
        tau_1 = residence_time(volume=100, volumetric_flow_rate=10)
        tau_2 = residence_time(volume=200, volumetric_flow_rate=10)
        assert tau_2 == pytest.approx(2 * tau_1)

    def test_residence_time_inverse_scaling(self):
        """Doubling flow rate halves residence time (same volume)"""
        tau_1 = residence_time(volume=100, volumetric_flow_rate=10)
        tau_2 = residence_time(volume=100, volumetric_flow_rate=20)
        assert tau_2 == pytest.approx(tau_1 / 2)

    def test_residence_time_design_scenario(self):
        """
        Reactor design problem:
        For first-order reaction with k=0.1 s⁻¹, to achieve 95% conversion:
        X = k*τ / (1 + k*τ) [CSTR conversion]
        0.95 = 0.1*τ / (1 + 0.1*τ) → τ ≈ 19 seconds
        
        Required reactor volume if flow is 5 L/s:
        V = τ * V̇ = 19 * 5 = 95 L
        """
        tau_required = 19  # seconds (calculated from kinetics)
        flow_rate = 5  # L/s
        volume_required = tau_required * flow_rate
        tau_calculated = residence_time(volume_required, flow_rate)
        assert tau_calculated == pytest.approx(tau_required)

    def test_residence_time_zero_volume(self):
        """Error: zero reactor volume (impossible)"""
        with pytest.raises(ValueError, match="volume must be positive"):
            residence_time(volume=0.0, volumetric_flow_rate=10)

    def test_residence_time_negative_volume(self):
        """Error: negative reactor volume"""
        with pytest.raises(ValueError, match="volume must be positive"):
            residence_time(volume=-100, volumetric_flow_rate=10)

    def test_residence_time_zero_flow(self):
        """Error: zero flow rate (reactor at stasis)"""
        with pytest.raises(ValueError, match="volumetric_flow_rate must be positive"):
            residence_time(volume=100, volumetric_flow_rate=0.0)

    def test_residence_time_negative_flow(self):
        """Error: negative flow rate (backward flow)"""
        with pytest.raises(ValueError, match="volumetric_flow_rate must be positive"):
            residence_time(volume=100, volumetric_flow_rate=-5)


# ============================================================================
# Integration Tests: Real-World Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests combining multiple functions in realistic scenarios"""

    def test_scenario_pharmaceutical_shelf_life(self):
        """
        Scenario: Drug stability study.
        Active pharmaceutical ingredient (API) degradation follows first-order kinetics.
        Storage temperature: 25°C
        Pre-exponential factor A: 1e10 hr⁻¹
        Activation energy: 80 kJ/mol
        Initial concentration: 100 mg/L
        Acceptable level: 90 mg/L (10% degradation)
        
        Find: Shelf life at 25°C
        """
        A = 1e10  # hr⁻¹
        Ea = 80000  # J/mol
        T = 25 + 273.15  # K
        
        # Calculate rate constant at 25°C
        k_hr = arrhenius_rate_constant(A, Ea, T)  # hr⁻¹
        k_s = k_hr / 3600  # Convert to s⁻¹
        
        # Time to reach 90 mg/L from 100 mg/L
        C0 = 100
        C_target = 90
        # For first-order: C = C0 * exp(-kt)
        # 90 = 100 * exp(-k*t)
        # t = -ln(0.9) / k
        t_s = -np.log(0.9) / k_s
        t_days = t_s / (24 * 3600)
        
        assert t_days > 0  # Shelf life is positive

    def test_scenario_cstr_reactor_design(self):
        """
        Scenario: CSTR design for first-order reaction.
        Target conversion: 80%
        Reaction kinetics: k = 0.05 s⁻¹ (first-order)
        Feed flow rate: 2 L/s
        
        Find: Required reactor volume
        """
        X_target = 0.80
        k = 0.05  # s⁻¹
        Q = 2  # L/s
        
        # For CSTR: X = k*τ / (1 + k*τ)
        # Solving: τ = X / (k*(1-X))
        tau = X_target / (k * (1 - X_target))
        
        # Required volume
        V = residence_time_inverse = Q * tau
        # Verify using residence_time function
        tau_verify = residence_time(V, Q)
        assert tau_verify == pytest.approx(tau)

    def test_scenario_temperature_impact_reaction_rate(self):
        """
        Scenario: How does heating accelerate a reaction?
        Reaction: A → B (first-order)
        Initial conditions: T₁ = 20°C, k₁ = 0.001 s⁻¹
        Target conditions: T₂ = 30°C
        
        Estimate new rate constant using Arrhenius equation.
        """
        # Assume Ea = 60 kJ/mol (typical for many reactions)
        A = 0.001 * np.exp(60000 / (8.314 * 293.15))  # Back-calculate A from k at 293.15 K
        Ea = 60000
        
        k_20C = arrhenius_rate_constant(A, Ea, 293.15)
        k_30C = arrhenius_rate_constant(A, Ea, 303.15)
        
        # Verify rate increase
        assert k_30C > k_20C
        assert k_30C / k_20C == pytest.approx(2.0, rel=0.3)  # Typical 2-3× increase

    def test_scenario_conversion_versus_residence_time(self):
        """
        Scenario: How does residence time affect conversion?
        Compare conversions at different residence times for first-order reaction.
        k = 0.1 s⁻¹, initial concentration = 1 mol/L
        """
        k = 0.1
        C0 = 1.0
        
        # At tau = 5 s
        c_5s = first_order_concentration(C0, k, 5)
        X_5s = conversion(C0, c_5s)
        
        # At tau = 10 s
        c_10s = first_order_concentration(C0, k, 10)
        X_10s = conversion(C0, c_10s)
        
        # At tau = 20 s
        c_20s = first_order_concentration(C0, k, 20)
        X_20s = conversion(C0, c_20s)
        
        # Verify monotonic increase in conversion
        assert X_5s < X_10s < X_20s
        assert X_20s < 1.0  # Never complete in first-order (asymptotic)
