def summarize_dataframe(dataframe):
    """
    Generate a structured summary of a pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to summarize.

    Returns
    -------
    dict
        A dictionary containing structural information about the DataFrame.
    """
    num_rows = len(dataframe)

    num_columns = len(dataframe.columns)

    column_names = list(dataframe.columns)

    data_types = {str(col): str(dtype) for col, dtype in dataframe.dtypes.items()}

    memory_usage_bytes = dataframe.memory_usage(deep=True).sum()
    memory_usage_mb = round(memory_usage_bytes / (1024 * 1024), 2)

    numerical_columns = dataframe.select_dtypes(include=["number"]).shape[1]

    categorical_columns = dataframe.select_dtypes(include=["object", "category"]).shape[1]

    summary = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "column_names": column_names,
        "data_types": data_types,
        "memory_usage_mb": memory_usage_mb,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }

    return summary
