import pandas as pd
import numpy as np
from scipy import ndimage


def moving_average(series, window_size):
    """
    Compute the moving average of a signal using a rolling window.

    Parameters
    ----------
    series : pandas.Series
        The input signal to smooth.
    window_size : int
        The size of the rolling window (number of points to average).

    Returns
    -------
    pandas.Series
        The smoothed signal where each value is the average of window_size points.

    Raises
    ------
    ValueError
        If window_size is not a positive integer or is larger than the series length.
    TypeError
        If series is not a pandas Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(window_size, int):
        raise ValueError("Window size must be an integer.")

    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    if window_size > len(series):
        raise ValueError(
            f"Window size ({window_size}) cannot be larger than series length ({len(series)})."
        )

    smoothed = series.rolling(window=window_size, center=True).mean()

    return smoothed


def rolling_standard_deviation(series, window_size):
    """
    Compute the rolling standard deviation of a signal.

    Parameters
    ----------
    series : pandas.Series
        The input signal to analyze.
    window_size : int
        The size of the rolling window (number of points per calculation).

    Returns
    -------
    pandas.Series
        The rolling standard deviation where each value represents the variation
        within the corresponding window.

    Raises
    ------
    ValueError
        If window_size is not a positive integer or is larger than the series length.
    TypeError
        If series is not a pandas Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(window_size, int):
        raise ValueError("Window size must be an integer.")

    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    if window_size > len(series):
        raise ValueError(
            f"Window size ({window_size}) cannot be larger than series length ({len(series)})."
        )

    rolling_std = series.rolling(window=window_size, center=True).std()

    return rolling_std


def smooth_signal(series, window_size):
    """
    Smooth a signal using a uniform filter from SciPy.

    This function applies a more aggressive smoothing than moving average,
    useful for highly noisy sensor data.

    Parameters
    ----------
    series : pandas.Series
        The input signal to smooth.
    window_size : int
        The size of the smoothing window (number of points to include).

    Returns
    -------
    pandas.Series
        The smoothed signal with reduced high-frequency noise.

    Raises
    ------
    ValueError
        If window_size is not a positive integer or is larger than the series length.
    TypeError
        If series is not a pandas Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(window_size, int):
        raise ValueError("Window size must be an integer.")

    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    if window_size > len(series):
        raise ValueError(
            f"Window size ({window_size}) cannot be larger than series length ({len(series)})."
        )

    series_values = series.values

    smoothed_values = ndimage.uniform_filter1d(series_values, size=window_size, mode='nearest')

    smoothed_series = pd.Series(smoothed_values, index=series.index, name=series.name)

    return smoothed_series
