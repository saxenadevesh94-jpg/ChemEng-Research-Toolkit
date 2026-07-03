import pandas as pd


def compute_descriptive_statistics(dataframe):
    """
    Compute descriptive statistics for numerical columns in a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    dict
        A dictionary where each key is a numerical column name and each value
        is a dictionary containing descriptive statistics.
    """
    numerical_data = dataframe.select_dtypes(include=["number"])

    if numerical_data.shape[1] == 0:
        return {}

    statistics = {}

    for column in numerical_data.columns:
        column_data = numerical_data[column]

        mean_value = column_data.mean()

        median_value = column_data.median()

        std_value = column_data.std()

        min_value = column_data.min()

        max_value = column_data.max()

        q1_value = column_data.quantile(0.25)

        q3_value = column_data.quantile(0.75)

        statistics[str(column)] = {
            "mean": round(float(mean_value), 4) if not pd.isna(mean_value) else None,
            "median": round(float(median_value), 4) if not pd.isna(median_value) else None,
            "std": round(float(std_value), 4) if not pd.isna(std_value) else None,
            "min": round(float(min_value), 4) if not pd.isna(min_value) else None,
            "max": round(float(max_value), 4) if not pd.isna(max_value) else None,
            "q1": round(float(q1_value), 4) if not pd.isna(q1_value) else None,
            "q3": round(float(q3_value), 4) if not pd.isna(q3_value) else None,
        }

    return statistics
