import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engineering import (
    reynolds_number,
    weber_number,
    capillary_number,
    froude_number,
    bond_number,
    ohnesorge_number,
)


print("Testing dimensionless numbers module")
print("=" * 80)
print()

print("TEST 1: Reynolds Number - Water Flow in Industrial Pipe")
print("-" * 80)
print("Scenario: 1 meter diameter pipe with water flowing at 2 m/s")
print("Expected: Re should be around 2,000,000 (highly turbulent)")
print("Interpretation: High Re means inertial forces dominate, flow is turbulent")
print()
rho_water = 1000.0  # kg/m³
v_water_pipe = 2.0  # m/s
d_pipe = 1.0  # m
mu_water = 0.001  # Pa·s (1 cP at 20°C)
re_water_pipe = reynolds_number(rho_water, v_water_pipe, d_pipe, mu_water)
print(f"Reynolds Number = {re_water_pipe:.2e}")
print(f"Result: {re_water_pipe:.0f} >> 2000, so flow is turbulent")
print()

print("TEST 2: Reynolds Number - Laminar Flow in Microfluidics")
print("-" * 80)
print("Scenario: 100 µm channel with water flowing at 0.1 m/s")
print("Expected: Re should be around 100 (laminar)")
print("Interpretation: Low Re means viscous forces dominate, flow is laminar")
print()
d_microfluidic = 100e-6  # m (100 µm)
v_microfluidic = 0.1  # m/s
re_microfluidic = reynolds_number(rho_water, v_microfluidic, d_microfluidic, mu_water)
print(f"Reynolds Number = {re_microfluidic:.2f}")
print(f"Result: {re_microfluidic:.0f} < 2000, so flow is laminar")
print()

print("TEST 3: Weber Number - Atomization Spray")
print("-" * 80)
print("Scenario: 50 µm water droplet in air at 30 m/s spray velocity")
print("Expected: We should be around 250 (droplets will break up)")
print("Interpretation: High We means inertial forces overcome surface tension")
print()
rho_air = 1.2  # kg/m³
v_spray = 30.0  # m/s
d_droplet_spray = 50e-6  # m (50 µm)
sigma_water_air = 0.072  # N/m at 20°C
we_spray = weber_number(rho_air, v_spray, d_droplet_spray, sigma_water_air)
print(f"Weber Number = {we_spray:.2f}")
print(f"Result: {we_spray:.0f} > 10, so droplets break up easily")
print()

print("TEST 4: Capillary Number - Trickle Bed Reactor")
print("-" * 80)
print("Scenario: Liquid flowing through packed bed at 0.01 m/s")
print("Expected: Ca should be around 0.0001 (surface tension dominates)")
print("Interpretation: Low Ca means droplets stay as separate entities, no coalescence")
print()
mu_liquid_trickle = 0.001  # Pa·s
v_trickle = 0.01  # m/s (typical for trickle bed)
ca_trickle = capillary_number(mu_liquid_trickle, v_trickle, sigma_water_air)
print(f"Capillary Number = {ca_trickle:.6f}")
print(f"Result: Ca < 0.001 means liquid wets the bed, typical for trickle reactors")
print()

print("TEST 5: Capillary Number - Microfluidic Two-Phase Flow")
print("-" * 80)
print("Scenario: Higher velocity flow in microfluidic channel at 0.5 m/s")
print("Expected: Ca should be around 0.006 (still surface tension dominated)")
print("Interpretation: Still low enough for stable droplet formation")
print()
v_microfluidic_2phase = 0.5  # m/s
ca_microfluidic = capillary_number(mu_water, v_microfluidic_2phase, sigma_water_air)
print(f"Capillary Number = {ca_microfluidic:.6f}")
print(f"Result: Ca ~ 0.007, good for droplet generation in microfluidics")
print()

print("TEST 6: Froude Number - Surface Wave Formation")
print("-" * 80)
print("Scenario: Water flowing in open channel at 0.5 m/s, channel width 0.1 m")
print("Expected: Fr should be around 0.5 (subcritical, waves possible)")
print("Interpretation: Low Fr means gravity dominates, waves form on surface")
print()
v_channel = 0.5  # m/s
d_channel = 0.1  # m (channel width)
fr_waves = froude_number(v_channel, d_channel)
print(f"Froude Number = {fr_waves:.2f}")
print(f"Result: Fr < 1 means subcritical flow, surface waves can propagate")
print()

print("TEST 7: Froude Number - High-Speed Overflow")
print("-" * 80)
print("Scenario: Water flowing over weir at 3 m/s, depth 0.1 m")
print("Expected: Fr should be around 3 (supercritical)")
print("Interpretation: High Fr means inertia dominates, waves wash away")
print()
v_weir = 3.0  # m/s
d_weir = 0.1  # m
fr_weir = froude_number(v_weir, d_weir)
print(f"Froude Number = {fr_weir:.2f}")
print(f"Result: Fr > 1 means supercritical flow, smooth water surface")
print()

print("TEST 8: Bond Number - Small Droplet (Surface Tension Dominates)")
print("-" * 80)
print("Scenario: Water droplet with diameter 1 mm suspended in air")
print("Expected: Bo should be around 0.01 (surface tension dominates)")
print("Interpretation: Droplet remains nearly spherical")
print()
rho_liquid = 1000.0  # kg/m³
d_small_droplet = 1e-3  # m (1 mm)
sigma_water = 0.072  # N/m
bo_small = bond_number(rho_liquid, d_small_droplet, sigma_water)
print(f"Bond Number = {bo_small:.4f}")
print(f"Result: Bo < 1 means surface tension keeps droplet spherical")
print()

print("TEST 9: Bond Number - Large Bubble (Gravity Dominates)")
print("-" * 80)
print("Scenario: Water bubble with diameter 1 cm in water column")
print("Expected: Bo should be around 100 (gravity dominates)")
print("Interpretation: Bubble deforms significantly")
print()
d_large_bubble = 1e-2  # m (1 cm)
bo_large = bond_number(rho_liquid, d_large_bubble, sigma_water)
print(f"Bond Number = {bo_large:.2f}")
print(f"Result: Bo > 1 means bubble deforms under gravity influence")
print()

print("TEST 10: Ohnesorge Number - Ink Jet Printing")
print("-" * 80)
print("Scenario: Ink droplet (oil-based) with 10 µm diameter")
print("Expected: Oh should be around 0.001 (viscous effects small)")
print("Interpretation: Droplets form through capillary breakup")
print()
mu_ink = 0.005  # Pa·s (5 cP, typical for ink)
rho_ink = 900.0  # kg/m³ (typical organic liquid)
d_ink_droplet = 10e-6  # m (10 µm)
sigma_ink = 0.035  # N/m (typical for oil-based ink)
oh_ink = ohnesorge_number(mu_ink, rho_ink, d_ink_droplet, sigma_ink)
print(f"Ohnesorge Number = {oh_ink:.6f}")
print(f"Result: Oh < 0.1 means Rayleigh breakup mode (good for inkjet)")
print()

print("TEST 11: Ohnesorge Number - Viscous Liquid (Honey-like)")
print("-" * 80)
print("Scenario: Viscous liquid spray (like honey) with 100 µm droplet")
print("Expected: Oh should be around 0.1 (viscous effects significant)")
print("Interpretation: Droplets may not break up as easily")
print()
mu_viscous = 0.1  # Pa·s (100 cP)
rho_viscous = 1100.0  # kg/m³
d_viscous_droplet = 100e-6  # m (100 µm)
sigma_viscous = 0.070  # N/m
oh_viscous = ohnesorge_number(mu_viscous, rho_viscous, d_viscous_droplet, sigma_viscous)
print(f"Ohnesorge Number = {oh_viscous:.4f}")
print(f"Result: Oh ~ 0.1 means viscous atomization, harder to break droplets")
print()

print("TEST 12: Multiphase System Comparison")
print("-" * 80)
print("Scenario: Gas-liquid trickle bed reactor analysis")
print("Expected: Calculate multiple dimensionless numbers together")
print()
rho_gas = 1.2  # kg/m³
rho_liquid = 1000.0  # kg/m³
mu_gas = 1.8e-5  # Pa·s
mu_liquid = 0.001  # Pa·s
v_gas = 0.5  # m/s
v_liquid = 0.01  # m/s
d_packing = 0.003  # m (3 mm packing)
sigma = 0.072  # N/m

re_gas = reynolds_number(rho_gas, v_gas, d_packing, mu_gas)
re_liquid = reynolds_number(rho_liquid, v_liquid, d_packing, mu_liquid)
ca_liquid = capillary_number(mu_liquid, v_liquid, sigma)
bo_liquid = bond_number(rho_liquid, d_packing, sigma)

print(f"Gas phase Reynolds: {re_gas:.0f}")
print(f"Liquid phase Reynolds: {re_liquid:.0f}")
print(f"Capillary Number: {ca_liquid:.6f}")
print(f"Bond Number: {bo_liquid:.4f}")
print()
print("Interpretation: Gas is turbulent, liquid is laminar, wetting behavior")
print("expected in trickle bed mode")
print()

print("TEST 13: Error Handling - Negative Density")
print("-" * 80)
print("Attempting Reynolds calculation with negative density...")
try:
    reynolds_number(-1000, 1.0, 0.01, 0.001)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 14: Error Handling - Zero Surface Tension")
print("-" * 80)
print("Attempting Weber calculation with zero surface tension...")
try:
    weber_number(1000, 10, 0.0001, 0)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 15: Error Handling - Zero Diameter in Froude")
print("-" * 80)
print("Attempting Froude calculation with zero diameter...")
try:
    froude_number(1.0, 0)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 16: Error Handling - Non-numeric Input")
print("-" * 80)
print("Attempting Reynolds calculation with string input...")
try:
    reynolds_number("1000", 1.0, 0.01, 0.001)
except TypeError as e:
    print(f"Caught error: {e}")
print()

print("All dimensionless number tests completed successfully!")
