import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.analysis import analyze_missing_values


print("Testing data quality module")
print("=" * 40)
print()

complete_df = pd.DataFrame(
    {
        "temperature": [25.3, 26.1, 24.8],
        "pressure": [101.3, 101.2, 101.1],
        "time": [1, 2, 3],
    }
)
print("1. Complete data (no missing values)")
print(analyze_missing_values(complete_df))
print()

some_missing_df = pd.DataFrame(
    {
        "experiment_id": ["Exp_001", "Exp_002", np.nan],
        "temperature": [25.3, np.nan, 24.8],
        "pressure": [101.3, 101.2, 101.1],
    }
)
print("2. Data with some missing values")
print(analyze_missing_values(some_missing_df))
print()

all_missing_column_df = pd.DataFrame(
    {
        "column_a": [1, 2, 3],
        "column_b": [np.nan, np.nan, np.nan],
        "column_c": [4, 5, 6],
    }
)
print("3. DataFrame with one completely missing column")
print(analyze_missing_values(all_missing_column_df))
print()

sparse_df = pd.DataFrame(
    {
        "value_1": [10, np.nan, np.nan, np.nan],
        "value_2": [np.nan, 20, np.nan, np.nan],
        "value_3": [np.nan, np.nan, 30, np.nan],
    }
)
print("4. Sparse data (mostly missing)")
print(analyze_missing_values(sparse_df))
print()

single_row_df = pd.DataFrame(
    {
        "col_a": [1],
        "col_b": [np.nan],
    }
)
print("5. Single row with missing value")
print(analyze_missing_values(single_row_df))
print()

empty_df = pd.DataFrame()
print("6. Empty DataFrame")
print(analyze_missing_values(empty_df))
