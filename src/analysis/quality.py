def analyze_missing_values(dataframe):
    """
    Analyze missing values in a pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    dict
        A dictionary containing missing value information.
    """
    total_cells = dataframe.shape[0] * dataframe.shape[1]

    missing_per_column = dataframe.isnull().sum().to_dict()

    total_missing = sum(missing_per_column.values())

    percentage_missing_per_column = {}
    for col, missing_count in missing_per_column.items():
        if dataframe.shape[0] > 0:
            percentage = round((missing_count / dataframe.shape[0]) * 100, 2)
        else:
            percentage = 0.0
        percentage_missing_per_column[col] = percentage

    columns_with_missing = [col for col, count in missing_per_column.items() if count > 0]

    overall_percentage_missing = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    quality_report = {
        "total_missing": total_missing,
        "missing_per_column": missing_per_column,
        "percentage_missing_per_column": percentage_missing_per_column,
        "columns_with_missing": columns_with_missing,
        "overall_percentage_missing": overall_percentage_missing,
    }

    return quality_report
