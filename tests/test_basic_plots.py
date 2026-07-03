import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.visualization import (
    create_histogram,
    create_scatter_plot,
    create_line_plot,
    create_box_plot,
    create_bar_plot,
)


print("Testing basic plotting module")
print("=" * 60)
print()

print("TEST 1: Histogram")
print("-" * 60)
print("Description: Shows the distribution of reaction temperatures")
print("Expected: Bell curve centered around 25°C")
print()
temperature_data = pd.DataFrame(
    {
        "temperature": np.random.normal(loc=25, scale=2, size=100),
    }
)
fig, ax = create_histogram(temperature_data, "temperature")
plt.show()
print()

print("TEST 2: Scatter Plot")
print("-" * 60)
print("Description: Shows relationship between temperature and yield")
print("Expected: Positive correlation (higher temp = higher yield)")
print()
scatter_data = pd.DataFrame(
    {
        "temperature": [20, 22, 25, 28, 30, 32, 35, 38, 40, 42],
        "yield": [0.45, 0.52, 0.65, 0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92],
    }
)
fig, ax = create_scatter_plot(scatter_data, "temperature", "yield")
plt.show()
print()

print("TEST 3: Line Plot")
print("-" * 60)
print("Description: Shows how pressure changes over time during reaction")
print("Expected: Increasing pressure trend over 60 seconds")
print()
time_series_data = pd.DataFrame(
    {
        "time_seconds": list(range(0, 61, 10)),
        "pressure_atm": [1.0, 1.2, 1.5, 1.8, 2.1, 2.3, 2.5],
    }
)
fig, ax = create_line_plot(time_series_data, "time_seconds", "pressure_atm")
plt.show()
print()

print("TEST 4: Box Plot")
print("-" * 60)
print("Description: Shows distribution and outliers in viscosity measurements")
print("Expected: Box showing median and quartiles, with any outliers as points")
print()
viscosity_data = pd.DataFrame(
    {
        "viscosity": [10.2, 10.5, 10.8, 11.0, 11.2, 11.5, 12.0, 12.3, 25.0],
    }
)
fig, ax = create_box_plot(viscosity_data, "viscosity")
plt.show()
print()

print("TEST 5: Bar Plot")
print("-" * 60)
print("Description: Shows average yield for different chemical treatments")
print("Expected: Bars showing yield differences between treatments")
print()
bar_data = pd.DataFrame(
    {
        "treatment": ["Control", "Treatment_A", "Treatment_B", "Treatment_C"],
        "average_yield": [0.65, 0.78, 0.82, 0.75],
    }
)
fig, ax = create_bar_plot(bar_data, "treatment", "average_yield")
plt.show()
print()

print("TEST 6: Error Handling - Missing Column")
print("-" * 60)
print("Attempting to create histogram with non-existent column...")
try:
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    fig, ax = create_histogram(df, "col_b")
except ValueError as e:
    print(f"Caught error: {e}")
print()

print("TEST 7: Error Handling - Non-numerical Column")
print("-" * 60)
print("Attempting to create histogram with text column...")
try:
    df = pd.DataFrame({"experiment": ["Exp_A", "Exp_B", "Exp_C"]})
    fig, ax = create_histogram(df, "experiment")
except TypeError as e:
    print(f"Caught error: {e}")
print()

print("All tests completed successfully!")
