import pandas as pd

from src.analysis import summarize_dataframe


print("Testing dataset summary module")
print("=" * 40)
print()

numerical_df = pd.DataFrame(
    {
        "temperature": [25.3, 26.1, 24.8],
        "pressure": [101.3, 101.2, 101.1],
        "time": [1, 2, 3],
    }
)
print("1. Numerical data")
print(summarize_dataframe(numerical_df))
print()

mixed_df = pd.DataFrame(
    {
        "experiment_id": ["Exp_001", "Exp_002", "Exp_003"],
        "temperature": [25.3, 26.1, 24.8],
        "chemical": ["NaCl", "KCl", "NaCl"],
        "pressure": [101.3, 101.2, 101.1],
    }
)
print("2. Mixed numerical and categorical data")
print(summarize_dataframe(mixed_df))
print()

large_df = pd.DataFrame(
    {f"column_{i}": range(1000) for i in range(10)}
)
print("3. Large DataFrame (1000 rows, 10 columns)")
summary = summarize_dataframe(large_df)
print(f"   Rows: {summary['num_rows']}")
print(f"   Columns: {summary['num_columns']}")
print(f"   Memory usage: {summary['memory_usage_mb']} MB")
print()

single_row_df = pd.DataFrame(
    {
        "value_1": [100],
        "value_2": [200],
    }
)
print("4. Single row DataFrame")
print(summarize_dataframe(single_row_df))
print()

empty_df = pd.DataFrame()
print("5. Empty DataFrame")
print(summarize_dataframe(empty_df))
