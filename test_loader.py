import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.io import load_csv_file


sample_file = Path("data/sample_data.csv")

try:
    dataframe = load_csv_file(sample_file)
    print("Success: CSV file loaded successfully.")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print("First five rows:")
    print(dataframe.head())
except FileNotFoundError as exc:
    print(f"Error: {exc}")
except ValueError as exc:
    print(f"Error: {exc}")
except Exception as exc:
    print(f"Unexpected error: {exc}")
