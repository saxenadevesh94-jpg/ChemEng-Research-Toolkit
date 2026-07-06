import pandas as pd
import numpy as np
from scipy import ndimage
from scipy import signal


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


def moving_average_filter(data, window_size):
    """
    Apply a moving average filter to a signal using NumPy arrays.

    The moving average filter replaces each point with the average of itself
    and its neighbors. It reduces noise uniformly but blurs sharp features.

    Parameters
    ----------
    data : numpy.ndarray
        The input signal as a 1D NumPy array.
    window_size : int
        The size of the moving window (number of points to average).
        Must be odd for symmetric centering.

    Returns
    -------
    numpy.ndarray
        The filtered signal with the same shape as input.

    Raises
    ------
    ValueError
        If window_size is not positive, not an integer, or larger than data length.
    TypeError
        If data is not a NumPy array or window_size is not numeric.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")

    if not isinstance(window_size, int):
        raise ValueError("Window size must be an integer.")

    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    if window_size > len(data):
        raise ValueError(
            f"Window size ({window_size}) cannot be larger than data length ({len(data)})."
        )

    kernel = np.ones(window_size) / window_size

    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data


def median_filter(data, window_size):
    """
    Apply a median filter to a signal using NumPy/SciPy.

    The median filter replaces each point with the median of its neighbors.
    It is excellent for removing outliers and salt-and-pepper noise while
    preserving sharp edges better than moving average.

    Parameters
    ----------
    data : numpy.ndarray
        The input signal as a 1D NumPy array.
    window_size : int
        The size of the median window (number of points to consider).
        Should typically be odd.

    Returns
    -------
    numpy.ndarray
        The filtered signal with the same shape as input.

    Raises
    ------
    ValueError
        If window_size is not positive, not an integer, or larger than data length.
    TypeError
        If data is not a NumPy array or window_size is not numeric.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")

    if not isinstance(window_size, int):
        raise ValueError("Window size must be an integer.")

    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    if window_size > len(data):
        raise ValueError(
            f"Window size ({window_size}) cannot be larger than data length ({len(data)})."
        )

    filtered_data = ndimage.median_filter(data, size=window_size, mode='nearest')

    return filtered_data


def savitzky_golay_filter(data, window_length, polyorder):
    """
    Apply a Savitzky-Golay filter to a signal.

    The Savitzky-Golay filter fits a polynomial to sliding windows of data.
    It smooths the signal while preserving peaks, edges, and derivatives.
    Superior to moving average for preserving features.

    Parameters
    ----------
    data : numpy.ndarray
        The input signal as a 1D NumPy array.
    window_length : int
        The length of the filter window (must be odd and at least polyorder + 1).
    polyorder : int
        The polynomial order (degree) to fit (typically 2-5).

    Returns
    -------
    numpy.ndarray
        The filtered signal with the same shape as input.

    Raises
    ------
    ValueError
        If window_length is invalid, polyorder is invalid, or conditions not met.
    TypeError
        If data is not a NumPy array or parameters are not numeric.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")

    if not isinstance(window_length, int):
        raise ValueError("Window length must be an integer.")

    if not isinstance(polyorder, int):
        raise ValueError("Polynomial order must be an integer.")

    if window_length < 1:
        raise ValueError("Window length must be at least 1.")

    if window_length % 2 == 0:
        raise ValueError("Window length must be odd.")

    if polyorder < 0:
        raise ValueError("Polynomial order cannot be negative.")

    if polyorder >= window_length:
        raise ValueError(
            f"Polynomial order ({polyorder}) must be less than window length ({window_length})."
        )

    if window_length > len(data):
        raise ValueError(
            f"Window length ({window_length}) cannot be larger than data length ({len(data)})."
        )

    filtered_data = signal.savgol_filter(data, window_length, polyorder, mode='nearest')

    return filtered_data
