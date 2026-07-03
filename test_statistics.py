import pandas as pd
import numpy as np

from src.analysis import compute_descriptive_statistics


print("Testing descriptive statistics module")
print("=" * 50)
print()

numerical_df = pd.DataFrame(
    {
        "temperature": [20.0, 21.5, 22.3, 19.8, 21.1],
        "pressure": [100.2, 101.1, 99.5, 100.8, 101.5],
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)
print("1. Numerical dataset")
print(compute_descriptive_statistics(numerical_df))
print()

mixed_df = pd.DataFrame(
    {
        "experiment": ["Exp_A", "Exp_B", "Exp_C", "Exp_D"],
        "temperature": [20.5, 21.2, 19.8, 22.1],
        "chemical": ["NaCl", "KCl", "NaCl", "KCl"],
        "yield": [0.78, 0.82, 0.75, 0.81],
    }
)
print("2. Mixed numerical and categorical dataset")
print(compute_descriptive_statistics(mixed_df))
print()

missing_values_df = pd.DataFrame(
    {
        "measurement_1": [10.0, np.nan, 12.5, 11.2, np.nan],
        "measurement_2": [20.1, 21.3, np.nan, 19.8, 22.5],
        "measurement_3": [30.0, 31.0, 32.0, 33.0, 34.0],
    }
)
print("3. Dataset containing missing values")
print(compute_descriptive_statistics(missing_values_df))
print()

single_row_df = pd.DataFrame(
    {
        "value_1": [42.5],
        "value_2": [100.0],
        "value_3": [9.75],
    }
)
print("4. Single-row dataset")
print(compute_descriptive_statistics(single_row_df))
print()

empty_df = pd.DataFrame()
print("5. Empty DataFrame")
print(compute_descriptive_statistics(empty_df))
