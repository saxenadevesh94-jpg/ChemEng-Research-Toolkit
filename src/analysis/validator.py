from collections import Counter


def validate_dataframe(dataframe):
    """
    Validate a pandas DataFrame and return a beginner-friendly result report.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to validate.

    Returns
    -------
    dict
        A dictionary containing the validation status and a list of issues.
    """
    issues = []

    if dataframe is None:
        issues.append("The provided data is missing.")
        return {"is_valid": False, "issues": issues}

    if dataframe.empty:
        issues.append("The DataFrame is empty.")

    if dataframe.shape[1] == 0:
        issues.append("The DataFrame has no columns.")

    column_names = list(dataframe.columns)
    duplicate_columns = [name for name, count in Counter(column_names).items() if count > 1]
    if duplicate_columns:
        issues.append(
            "Duplicate column names found: " + ", ".join(map(str, duplicate_columns)) + "."
        )

    if dataframe.duplicated().any():
        issues.append("Duplicate rows were found.")

    unnamed_columns = [
        name for name in column_names if isinstance(name, str) and name.lower().startswith("unnamed")
    ]
    if unnamed_columns:
        issues.append("Unnamed columns were found.")

    return {"is_valid": len(issues) == 0, "issues": issues}
