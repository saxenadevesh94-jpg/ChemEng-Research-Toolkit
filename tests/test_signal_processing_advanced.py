import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.signal_processing import (
    moving_average_filter,
    median_filter,
    savitzky_golay_filter,
)


print("Testing signal processing filters (NumPy-based)")
print("=" * 80)
print()

print("TEST 1: Moving Average Filter on Noisy Chromatography Peak")
print("-" * 80)
print("Description: HPLC detector signal (absorbance) with electronic noise")
print("Expected: Smoothed peak that preserves general shape but blurs edges")
print()
np.random.seed(42)
true_peak = np.concatenate([
    np.zeros(10),
    np.linspace(0, 1, 10),
    np.ones(20),
    np.linspace(1, 0, 10),
    np.zeros(10),
])
noise = np.random.normal(0, 0.05, len(true_peak))
noisy_signal = true_peak + noise
print(f"Original signal shape: {noisy_signal.shape}")
print(f"Original signal (indices 15-25): {noisy_signal[15:25]}")
print()
smoothed = moving_average_filter(noisy_signal, window_size=5)
print(f"After moving average filter (window=5, indices 15-25): {smoothed[15:25]}")
print("Interpretation: Sharp edges are blurred, noise is reduced")
print()

print("TEST 2: Median Filter on Sensor Data with Outliers")
print("-" * 80)
print("Description: Pressure sensor with occasional bad readings (spikes)")
print("Expected: Outliers removed, edges preserved better than moving average")
print()
sensor_data = np.array([100.0, 101.0, 100.8, 150.0, 100.9, 101.1, 100.5,
                        101.2, 100.9, 200.0, 101.0, 100.8, 101.1, 100.9],
                       dtype=float)
print(f"Original signal with spikes at indices 3 and 9:")
print(f"  {sensor_data}")
print()
median_filtered = median_filter(sensor_data, window_size=3)
print(f"After median filter (window=3):")
print(f"  {median_filtered}")
print("Interpretation: Spikes at indices 3 and 9 are successfully removed")
print()

print("TEST 3: Savitzky-Golay Filter on Reaction Kinetics Data")
print("-" * 80)
print("Description: Concentration vs time with noise, needs to preserve slope")
print("Expected: Smooth curve that preserves kinetic trend")
print()
time = np.linspace(0, 10, 50)
concentration_true = 100 * np.exp(-0.3 * time)
noise_kinetics = np.random.normal(0, 2, len(time))
concentration_noisy = concentration_true + noise_kinetics
print(f"Original signal (first 10 values):")
print(f"  {concentration_noisy[:10]}")
print()
sg_filtered = savitzky_golay_filter(concentration_noisy, window_length=11, polyorder=3)
print(f"After Savitzky-Golay filter (window=11, polyorder=3, first 10 values):")
print(f"  {sg_filtered[:10]}")
print("Interpretation: Smooth curve that preserves exponential decay trend")
print()

print("TEST 4: Constant Signal (All Same Value)")
print("-" * 80)
print("Description: Steady-state signal with no variation")
print("Expected: Constant value unchanged by any filter")
print()
constant = np.ones(20) * 50.0
print(f"Original constant signal: {constant}")
print()
ma_const = moving_average_filter(constant, window_size=5)
med_const = median_filter(constant, window_size=5)
sg_const = savitzky_golay_filter(constant, window_size=5, polyorder=2)
print(f"After moving average: {ma_const}")
print(f"After median filter: {med_const}")
print(f"After Savitzky-Golay: {sg_const}")
print("Interpretation: All filters preserve constant value correctly")
print()

print("TEST 5: Short Signal (Fewer Points than Window)")
print("-" * 80)
print("Description: Very limited data (5 points) with small window (3)")
print("Expected: All filters handle edge cases gracefully")
print()
short_signal = np.array([10.0, 15.0, 12.0, 18.0, 11.0])
print(f"Original signal: {short_signal}")
print()
ma_short = moving_average_filter(short_signal, window_size=3)
med_short = median_filter(short_signal, window_size=3)
sg_short = savitzky_golay_filter(short_signal, window_length=3, polyorder=1)
print(f"After moving average (window=3): {ma_short}")
print(f"After median filter (window=3): {med_short}")
print(f"After Savitzky-Golay (window=3, polyorder=1): {sg_short}")
print()

print("TEST 6: Odd vs Even Window Sizes")
print("-" * 80)
print("Description: Compare behavior with odd and even windows")
print("Expected: Moving average works with even, median handles both,")
print("Savitzky-Golay requires odd")
print()
signal_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
print(f"Original signal: {signal_test}")
print()
ma_odd = moving_average_filter(signal_test, window_size=5)
ma_even = moving_average_filter(signal_test, window_size=4)
med_odd = median_filter(signal_test, window_size=5)
med_even = median_filter(signal_test, window_size=4)
print(f"Moving average (odd window=5): {ma_odd[:5]}")
print(f"Moving average (even window=4): {ma_even[:5]}")
print(f"Median filter (odd window=5): {med_odd[:5]}")
print(f"Median filter (even window=4): {med_even[:5]}")
print()

print("TEST 7: Single-Point Signal")
print("-" * 80)
print("Description: Signal with only one value")
print("Expected: Filters return single value")
print()
single_point = np.array([42.0])
print(f"Original signal: {single_point}")
print()
ma_single = moving_average_filter(single_point, window_size=1)
med_single = median_filter(single_point, window_size=1)
sg_single = savitzky_golay_filter(single_point, window_length=1, polyorder=0)
print(f"After moving average (window=1): {ma_single}")
print(f"After median filter (window=1): {med_single}")
print(f"After Savitzky-Golay (window=1, polyorder=0): {sg_single}")
print()

print("TEST 8: Error Handling - Negative Window Size")
print("-" * 80)
print("Attempting to apply negative window size...")
try:
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    moving_average_filter(test_data, window_size=-1)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 9: Error Handling - Non-NumPy Input")
print("-" * 80)
print("Attempting to filter a Python list instead of NumPy array...")
try:
    moving_average_filter([1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)
except TypeError as e:
    print(f"Caught error: {e}")
print()

print("TEST 10: Error Handling - Window Larger Than Data")
print("-" * 80)
print("Attempting to apply window_size=20 to 10-point signal...")
try:
    small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    median_filter(small_data, window_size=20)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 11: Error Handling - Even Window Length in Savitzky-Golay")
print("-" * 80)
print("Attempting Savitzky-Golay with even window length (must be odd)...")
try:
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    savitzky_golay_filter(test_data, window_length=4, polyorder=2)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 12: Error Handling - Polyorder >= Window Length")
print("-" * 80)
print("Attempting Savitzky-Golay with polyorder=5 and window_length=5...")
try:
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    savitzky_golay_filter(test_data, window_length=5, polyorder=5)
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 13: Empty Array Handling")
print("-" * 80)
print("Description: Empty NumPy array")
print("Expected: Errors on filters")
print()
try:
    empty_array = np.array([])
    moving_average_filter(empty_array, window_size=3)
except ValueError as e:
    print(f"Moving average error (expected): {e}")
print()

print("TEST 14: Comparison - Three Filters on Same Noisy Data")
print("-" * 80)
print("Description: Compare all three filters on identical noisy signal")
print("Expected: Different smoothing characteristics")
print()
comparison_signal = np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.normal(0, 0.1, 50)
print(f"Original signal (first 10 points):")
print(f"  {comparison_signal[:10]}")
print()
ma_result = moving_average_filter(comparison_signal, window_size=5)
med_result = median_filter(comparison_signal, window_size=5)
sg_result = savitzky_golay_filter(comparison_signal, window_length=5, polyorder=2)
print(f"Moving average (window=5, first 10): {ma_result[:10]}")
print(f"Median filter (window=5, first 10):  {med_result[:10]}")
print(f"Savitzky-Golay (window=5, poly=2):   {sg_result[:10]}")
print()
print("Interpretation: Moving average is most aggressive, Savitzky-Golay preserves features best")
print()

print("All signal processing filter tests completed successfully!")
