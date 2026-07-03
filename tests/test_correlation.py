import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.analysis import compute_correlation_matrix


print("Testing correlation analysis module")
print("=" * 60)
print()

print("TEST 1: Strong positive correlation")
print("-" * 60)
print("Expected: X and Y should have correlation close to 1.0")
print("Reason: As X increases, Y increases proportionally")
print()
positive_df = pd.DataFrame(
    {
        "x_values": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y_values": [2.0, 4.0, 6.0, 8.0, 10.0],
    }
)
result_positive = compute_correlation_matrix(positive_df)
print("Correlation matrix:")
print(result_positive)
print("Explanation: x_values and y_values have a perfect linear relationship (y = 2*x),")
print("so their correlation is 1.0.")
print()

print("TEST 2: Strong negative correlation")
print("-" * 60)
print("Expected: A and B should have correlation close to -1.0")
print("Reason: As A increases, B decreases proportionally")
print()
negative_df = pd.DataFrame(
    {
        "temperature": [20.0, 30.0, 40.0, 50.0],
        "viscosity": [100.0, 75.0, 50.0, 25.0],
    }
)
result_negative = compute_correlation_matrix(negative_df)
print("Correlation matrix:")
print(result_negative)
print("Explanation: As temperature increases, viscosity decreases in a linear pattern,")
print("resulting in a correlation of -1.0.")
print()

print("TEST 3: Nearly zero correlation")
print("-" * 60)
print("Expected: Variable1 and Variable2 should have correlation close to 0.0")
print("Reason: No clear linear relationship between the variables")
print()
zero_df = pd.DataFrame(
    {
        "variable_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "variable_2": [5.2, 3.1, 4.8, 2.9, 5.1],
    }
)
result_zero = compute_correlation_matrix(zero_df)
print("Correlation matrix:")
print(result_zero)
print("Explanation: variable_2 values do not follow a linear pattern relative to variable_1,")
print("so the correlation is near zero.")
print()

print("TEST 4: Mixed numerical and categorical data")
print("-" * 60)
print("Expected: Only numerical columns should appear in the correlation matrix")
print("Reason: Correlation is only meaningful for numerical data")
print()
mixed_df = pd.DataFrame(
    {
        "experiment_id": ["Exp_A", "Exp_B", "Exp_C", "Exp_D"],
        "measurement_1": [10.0, 20.0, 30.0, 40.0],
        "chemical_type": ["NaCl", "KCl", "NaCl", "KCl"],
        "measurement_2": [5.0, 10.0, 15.0, 20.0],
    }
)
result_mixed = compute_correlation_matrix(mixed_df)
print("Correlation matrix:")
print(result_mixed)
print("Explanation: Only measurement_1 and measurement_2 appear in the output.")
print("experiment_id and chemical_type are excluded because they are not numerical.")
print()

print("TEST 5: Dataset containing missing values")
print("-" * 60)
print("Expected: Correlation should still be computed, ignoring missing values")
print("Reason: pandas corr() automatically skips NaN values")
print()
missing_df = pd.DataFrame(
    {
        "sensor_a": [10.0, np.nan, 30.0, 40.0, 50.0],
        "sensor_b": [5.0, 10.0, np.nan, 20.0, 25.0],
    }
)
result_missing = compute_correlation_matrix(missing_df)
print("Correlation matrix:")
print(result_missing)
print("Explanation: Despite missing values, the correlation is computed from available data.")
print()

print("TEST 6: Single numerical column")
print("-" * 60)
print("Expected: Empty DataFrame returned")
print("Reason: Correlation requires at least two variables to compare")
print()
single_col_df = pd.DataFrame(
    {
        "single_value": [1.0, 2.0, 3.0, 4.0],
    }
)
result_single = compute_correlation_matrix(single_col_df)
print("Correlation matrix:")
print(result_single)
print("Explanation: Only one numerical column exists. Correlation needs at least two columns,")
print("so an empty DataFrame is returned.")
print()

print("TEST 7: Empty DataFrame")
print("-" * 60)
print("Expected: Empty DataFrame returned")
print("Reason: Cannot compute correlation without data")
print()
empty_df = pd.DataFrame()
result_empty = compute_correlation_matrix(empty_df)
print("Correlation matrix:")
print(result_empty)
print("Explanation: The DataFrame is empty, so correlation cannot be computed.")
