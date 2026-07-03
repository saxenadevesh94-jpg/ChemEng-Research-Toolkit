import matplotlib.pyplot as plt


def create_histogram(dataframe, column):
    """
    Create a histogram for a numerical column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the data to plot.
    column : str
        The name of the numerical column to plot.

    Returns
    -------
    tuple
        A tuple containing (figure, axes) matplotlib objects.

    Raises
    ------
    ValueError
        If the column does not exist in the DataFrame.
    TypeError
        If the column contains non-numerical data.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if not dataframe[column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{column}' must contain numerical data.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dataframe[column], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {column}')
    ax.grid(axis='y', alpha=0.3)

    return fig, ax


def create_scatter_plot(dataframe, x_column, y_column):
    """
    Create a scatter plot for two numerical columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the data to plot.
    x_column : str
        The name of the column for the x-axis.
    y_column : str
        The name of the column for the y-axis.

    Returns
    -------
    tuple
        A tuple containing (figure, axes) matplotlib objects.

    Raises
    ------
    ValueError
        If either column does not exist in the DataFrame.
    TypeError
        If either column contains non-numerical data.
    """
    if x_column not in dataframe.columns:
        raise ValueError(f"Column '{x_column}' does not exist in the DataFrame.")
    if y_column not in dataframe.columns:
        raise ValueError(f"Column '{y_column}' does not exist in the DataFrame.")

    if not dataframe[x_column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{x_column}' must contain numerical data.")
    if not dataframe[y_column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{y_column}' must contain numerical data.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dataframe[x_column], dataframe[y_column], alpha=0.6, s=50)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{x_column} vs {y_column}')
    ax.grid(alpha=0.3)

    return fig, ax


def create_line_plot(dataframe, x_column, y_column):
    """
    Create a line plot for two numerical columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the data to plot.
    x_column : str
        The name of the column for the x-axis.
    y_column : str
        The name of the column for the y-axis.

    Returns
    -------
    tuple
        A tuple containing (figure, axes) matplotlib objects.

    Raises
    ------
    ValueError
        If either column does not exist in the DataFrame.
    TypeError
        If either column contains non-numerical data.
    """
    if x_column not in dataframe.columns:
        raise ValueError(f"Column '{x_column}' does not exist in the DataFrame.")
    if y_column not in dataframe.columns:
        raise ValueError(f"Column '{y_column}' does not exist in the DataFrame.")

    if not dataframe[x_column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{x_column}' must contain numerical data.")
    if not dataframe[y_column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{y_column}' must contain numerical data.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataframe[x_column], dataframe[y_column], marker='o', linestyle='-', linewidth=2)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{y_column} over {x_column}')
    ax.grid(alpha=0.3)

    return fig, ax


def create_box_plot(dataframe, column):
    """
    Create a box plot for a numerical column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the data to plot.
    column : str
        The name of the numerical column to plot.

    Returns
    -------
    tuple
        A tuple containing (figure, axes) matplotlib objects.

    Raises
    ------
    ValueError
        If the column does not exist in the DataFrame.
    TypeError
        If the column contains non-numerical data.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if not dataframe[column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{column}' must contain numerical data.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(dataframe[column])
    ax.set_ylabel(column)
    ax.set_title(f'Box Plot of {column}')
    ax.grid(axis='y', alpha=0.3)

    return fig, ax


def create_bar_plot(dataframe, category_column, value_column):
    """
    Create a bar plot for categorical and numerical data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the data to plot.
    category_column : str
        The name of the categorical column for the x-axis.
    value_column : str
        The name of the numerical column for the y-axis.

    Returns
    -------
    tuple
        A tuple containing (figure, axes) matplotlib objects.

    Raises
    ------
    ValueError
        If either column does not exist in the DataFrame.
    TypeError
        If the value column contains non-numerical data.
    """
    if category_column not in dataframe.columns:
        raise ValueError(f"Column '{category_column}' does not exist in the DataFrame.")
    if value_column not in dataframe.columns:
        raise ValueError(f"Column '{value_column}' does not exist in the DataFrame.")

    if not dataframe[value_column].dtype.kind in 'biufc':
        raise TypeError(f"Column '{value_column}' must contain numerical data.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(dataframe[category_column], dataframe[value_column], alpha=0.7, edgecolor='black')
    ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    ax.set_title(f'{value_column} by {category_column}')
    ax.grid(axis='y', alpha=0.3)

    return fig, ax
