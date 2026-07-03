import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.analysis import validate_dataframe


print("Testing validator module")
print("-" * 30)

empty_df = pd.DataFrame()
print("1. Empty DataFrame")
print(validate_dataframe(empty_df))
print()

no_columns_df = pd.DataFrame(index=[0, 1])
print("2. DataFrame with no columns")
print(validate_dataframe(no_columns_df))
print()

duplicate_columns_df = pd.DataFrame([[1, 2]], columns=["A", "A"])
print("3. Duplicate column names")
print(validate_dataframe(duplicate_columns_df))
print()

duplicate_rows_df = pd.DataFrame([[1, 2], [1, 2]], columns=["A", "B"])
print("4. Duplicate rows")
print(validate_dataframe(duplicate_rows_df))
print()

unnamed_columns_df = pd.DataFrame([[1, 2]], columns=["Unnamed: 0", "value"])
print("5. Unnamed columns")
print(validate_dataframe(unnamed_columns_df))
print()

valid_df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
print("6. Valid DataFrame")
print(validate_dataframe(valid_df))
