import pandas as pd


def compute_correlation_matrix(dataframe):
    """
    Compute the Pearson correlation matrix for numerical columns in a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pandas.DataFrame
        A correlation matrix where rows and columns are numerical column names,
        and values represent Pearson correlation coefficients.
        Returns an empty DataFrame if fewer than two numerical columns exist.
    """
    numerical_data = dataframe.select_dtypes(include=["number"])

    if numerical_data.shape[1] < 2:
        return pd.DataFrame()

    correlation_matrix = numerical_data.corr(method="pearson")

    return correlation_matrix
