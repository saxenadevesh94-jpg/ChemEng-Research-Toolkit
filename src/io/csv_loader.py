from pathlib import Path

import pandas as pd


def load_csv_file(file_path):
    """
    Load a CSV file and return it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        The location of the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        The contents of the CSV file as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the provided file does not exist.
    ValueError
        If the file is not a CSV file or cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            "The CSV file was not found. Please check the file path and try again."
        )

    if not path.is_file():
        raise FileNotFoundError(
            "The provided path does not point to a file. Please provide a valid CSV file path."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "The selected file is not a CSV file. Please choose a file with a .csv extension."
        )

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "The CSV file is empty. Please provide a CSV file with data."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "The CSV file could not be read correctly. Please check the file format and try again."
        ) from exc
    except UnicodeError as exc:
        raise ValueError(
            "The CSV file could not be read because of a text encoding issue. Please save the file in a standard text format and try again."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "The CSV file could not be read. Please check the file and try again."
        ) from exc

    return dataframe
