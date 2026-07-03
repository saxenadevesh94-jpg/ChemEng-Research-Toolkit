import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.signal_processing import moving_average, rolling_standard_deviation, smooth_signal


print("Testing signal processing filters")
print("=" * 70)
print()

print("TEST 1: Moving Average on Noisy Pressure Sensor")
print("-" * 70)
print("Description: Pressure measurements with random noise around 101 atm")
print("Expected: Smoother curve centered around 101 atm with sharp edges preserved")
print()
np.random.seed(42)
time_steps = np.arange(0, 100, 1)
true_pressure = 101.0
noisy_pressure = pd.Series(
    true_pressure + np.random.normal(0, 0.8, len(time_steps)),
    name="pressure_atm"
)
print("Original signal (first 10 values):")
print(noisy_pressure.head(10))
print()
ma_pressure = moving_average(noisy_pressure, window_size=5)
print("Moving average (window=5, first 10 values):")
print(ma_pressure.head(10))
print()

print("TEST 2: Moving Average on Temperature RTD Curve")
print("-" * 70)
print("Description: Temperature increasing from 20°C to 100°C with sensor noise")
print("Expected: Smooth increasing curve that follows the trend")
print()
temperature_true = np.linspace(20, 100, 50)
temperature_noisy = pd.Series(
    temperature_true + np.random.normal(0, 1.5, len(temperature_true)),
    name="temperature_celsius"
)
print("Original signal (first 10 values):")
print(temperature_noisy.head(10))
print()
ma_temperature = moving_average(temperature_noisy, window_size=5)
print("Moving average (window=5, first 10 values):")
print(ma_temperature.head(10))
print()

print("TEST 3: Rolling Standard Deviation on Flow Rate")
print("-" * 70)
print("Description: Flow rate with pulsating pattern (varying noise levels)")
print("Expected: Standard deviation higher where noise is higher")
print()
flow_rate = pd.Series(
    np.concatenate([
        np.random.normal(50, 1, 30),   # Low noise phase
        np.random.normal(50, 5, 30),   # High noise phase
        np.random.normal(50, 1, 30),   # Low noise phase again
    ]),
    name="flow_rate_ml_per_min"
)
print("Original signal (first 10 values):")
print(flow_rate.head(10))
print()
rolling_std = rolling_standard_deviation(flow_rate, window_size=10)
print("Rolling standard deviation (window=10, first 15 values):")
print(rolling_std.head(15))
print()

print("TEST 4: Smooth Signal on Highly Noisy ERT Measurement")
print("-" * 70)
print("Description: ERT conductivity signal with very high noise")
print("Expected: Much smoother signal that reveals underlying pattern")
print()
ert_true = np.sin(np.linspace(0, 4*np.pi, 60)) * 10 + 50
ert_noisy = pd.Series(
    ert_true + np.random.normal(0, 3, len(ert_true)),
    name="ert_conductivity"
)
print("Original signal (first 10 values):")
print(ert_noisy.head(10))
print()
ert_smoothed = smooth_signal(ert_noisy, window_size=7)
print("Smoothed signal (window=7, first 10 values):")
print(ert_smoothed.head(10))
print()

print("TEST 5: Processing Constant Signal (No Variation)")
print("-" * 70)
print("Description: Steady-state pressure with no variation")
print("Expected: Constant values remain constant after filtering")
print()
constant_signal = pd.Series([50.0] * 20, name="constant_pressure")
print("Original signal:")
print(constant_signal.values)
print()
ma_constant = moving_average(constant_signal, window_size=5)
print("After moving average:")
print(ma_constant.values)
print()

print("TEST 6: Processing Short Signal")
print("-" * 70)
print("Description: Very short signal (only 5 points)")
print("Expected: Works correctly with small window size")
print()
short_signal = pd.Series([10.0, 12.0, 11.0, 13.0, 10.5], name="short_measurement")
print("Original signal:")
print(short_signal.values)
print()
ma_short = moving_average(short_signal, window_size=3)
print("Moving average (window=3):")
print(ma_short.values)
print()

print("TEST 7: Signal with Missing Values")
print("-" * 70)
print("Description: Pressure signal with NaN values representing sensor errors")
print("Expected: Filtering handles missing values gracefully")
print()
signal_with_nan = pd.Series(
    [100.0, 101.0, np.nan, 99.5, 100.5, np.nan, 101.2, 100.8],
    name="pressure_with_gaps"
)
print("Original signal:")
print(signal_with_nan.values)
print()
ma_with_nan = moving_average(signal_with_nan, window_size=3)
print("Moving average (window=3):")
print(ma_with_nan.values)
print()
rolling_std_nan = rolling_standard_deviation(signal_with_nan, window_size=3)
print("Rolling standard deviation (window=3):")
print(rolling_std_nan.values)
print()

print("TEST 8: Error Handling - Window Larger Than Series")
print("-" * 70)
print("Attempting to apply window_size=50 to 20-point signal...")
try:
    small_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    moving_average(small_series, window_size=50)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 9: Error Handling - Non-Series Input")
print("-" * 70)
print("Attempting to filter a list instead of Series...")
try:
    moving_average([1, 2, 3, 4, 5], window_size=3)
except TypeError as e:
    print(f"Caught error: {e}")
print()

print("TEST 10: Error Handling - Invalid Window Size")
print("-" * 70)
print("Attempting to apply negative window size...")
try:
    test_series = pd.Series([1, 2, 3, 4, 5])
    moving_average(test_series, window_size=-1)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("All signal processing tests completed successfully!")
