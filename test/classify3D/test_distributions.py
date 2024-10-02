"""
A module for testing the Distributions class.

Name
----
test_distributions.py

Dependencies
------------
    External
    --------
    pytest
        A library for writing and running tests.
    numpy
        A library for efficient numerical computation.
    pandas
        A library for data manipulation and analysis.
    plotly
        A library for making interactive web applications.

    Internal
    --------
    distributions
        Module that contains the Distributions class.

Attributes
----------
    Functions
    ---------
    mock_data()
        Fixture to provide mock data for testing.
    distributions()
        Fixture to initialize the Distributions class.
    hex_counts()
        Fixture for expected hex_counts result for testing.
    test_init()
        Test that the Distributions class initializes correctly.
    test_repr()
        Test that the __repr__ method works correctly.
    test_get_distributions()
        Test that _get_distributions returns the correct plots.
    test_create_bar_plot()
        Test that the _create_bar_plot method creates a bar plot.
    test_create_histogram()
        Test that the _create_histogram method creates a histogram.
    test_create_kde_plot()
        Test that the _create_kde_plot method creates a KDE plot.
    test_get_distributions()
        Test that _get_distributions returns the correct plots.
    test_call_notebook()
        Test that the __call__ method works with 'notebook' display.
    test_call_container()
        Test that the __call__ method works with 'container' display.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from plotly import graph_objects as go
from src.classify3D.distributions import Distributions


@pytest.fixture
def mock_data() -> np.ndarray:
    """
    Fixture to provide mock data for testing.

    Name
    ----
    mock_data

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

    Returns
    -------
    np.ndarray
        Mock dataset.

    Examples
    --------
    >>> mock_data()
    """
    return np.array(object=[0, 1, 1, 2, 2, 2, 3, 3, 3, 3])  # Mock dataset


@pytest.fixture
def distributions(mock_data: np.ndarray) -> Distributions:
    """
    Fixture to initialize the Distributions class.

    Name
    ----
    distributions

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_data : np.ndarray
        Mock dataset.

    Returns
    -------
    Distributions
        Initialized Distributions class.

    Examples
    --------
    >>> distributions()
    """
    return Distributions(y=mock_data)


@pytest.fixture
def hex_counts() -> pd.DataFrame:
    """
    Fixture for expected hex_counts result for testing.

    Name
    ----
    hex_counts

    Dependencies
    ------------
        External
        --------
        pandas
            Package for data manipulation and analysis.

    Returns
    -------
    pd.DataFrame
        Expected hex_counts result.

    Examples
    --------
    >>> hex_counts()
    Hex Code  Count
    0        0      0
    1        1      3
    2        2      6
    3        3     10
    """
    counts = (
        pd.Series(data=[0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        .value_counts()
        .reset_index()
    )
    counts.columns = ['Hex Code', 'Count']
    return counts


def test_init(distributions: Distributions) -> None:
    """
    Test that the Distributions class initializes correctly.

    Name
    ----
    test_init

    Dependencies
    ------------
        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    distributions : Distributions
        Mock of the Distributions class.
    """
    assert isinstance(distributions, Distributions)
    assert hasattr(distributions, 'bar_plot')
    assert hasattr(distributions, 'histogram')
    assert hasattr(distributions, 'kde')


def test_repr(distributions: Distributions) -> None:
    """
    Test that the __repr__ method works correctly.

    Name
    ----
    test_repr

    Dependencies
    ------------
        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    distributions : Distributions
        Mock of the Distributions class.
    """
    assert repr(distributions) == "Distributions()"


@patch("src.classify3D.distributions.go.Figure.update_layout")
def test_update_layout(
    mock_update_layout: MagicMock, distributions: Distributions
) -> None:
    """
    Test the _update_layout method.

    Name
    ----
    test_update_layout

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.
        plotly
            A library for creating interactive user interfaces.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_update_layout : MagicMock
        Mock of the go.Figure.update_layout function.
    distributions : Distributions
        Mock of the Distributions class.
    """
    fig = go.Figure()
    updated_fig = distributions._update_layout(
        fig=fig,
        title_text="Test Title",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
    )

    mock_update_layout.assert_called_once()
    assert isinstance(updated_fig, go.Figure)


@patch("src.classify3D.distributions.px.bar")
def test_create_bar_plot(
    mock_bar: MagicMock, distributions: MagicMock, hex_counts: pd.DataFrame
) -> None:
    """
    Test that the _create_bar_plot method creates a bar plot.

    Name
    ----
    test_create_bar_plot

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.
        pandas
            A library for data manipulation and analysis.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_bar : MagicMock
        Mock of the px.bar function.
    distributions : MagicMock
        Mock of the Distributions class.
    hex_counts : pd.DataFrame
        Mock data for testing.
    """
    distributions._create_bar_plot(
        hex_counts, 'Hex Code', 'Count', 'Test Title'
    )
    mock_bar.assert_called_once()


@patch("src.classify3D.distributions.px.histogram")
def test_create_histogram(
    mock_histogram: MagicMock,
    distributions: Distributions,
    hex_counts: pd.DataFrame,
) -> None:
    """
    Test that the _create_histogram method creates a histogram.

    Name
    ----
    test_create_histogram

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.
        pandas
            A library for data manipulation and analysis.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_histogram : MagicMock
        Mock of the px.histogram function.
    distributions : Distributions
        Mock of the Distributions class.
    hex_counts : pd.DataFrame
        Mock data for testing.
    """
    distributions._create_histogram(
        data=hex_counts, x_col='Hex Code', title='Test Title', nbins=64
    )
    mock_histogram.assert_called_once()


@patch("src.classify3D.distributions.go.Figure.add_trace")
@patch("src.classify3D.distributions.stats.gaussian_kde")
def test_create_kde_plot(
    mock_kde: MagicMock,
    mock_add_trace: MagicMock,
    distributions: Distributions,
    hex_counts: pd.DataFrame,
) -> None:
    """
    Test that the _create_kde_plot method creates a KDE plot.

    Name
    ----
    test_create_kde_plot

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.
        numpy
            A library for efficient numerical computation.
        pandas
            A library for data manipulation and analysis.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_kde : MagicMock
        Mock of the stats.gaussian_kde function.
    mock_add_trace : MagicMock
        Mock of the go.Figure.add_trace function.
    distributions : Distributions
        Module that contains the Distributions class.
    hex_counts : pd.DataFrame
        Mock data for testing.
    """
    mock_kde.return_value = lambda x: np.array(object=[0.1, 0.2, 0.3])
    distributions._create_kde_plot(data=hex_counts, title="KDE Test")
    mock_add_trace.assert_called_once()


@patch("src.classify3D.distributions.Distributions._create_bar_plot")
@patch("src.classify3D.distributions.Distributions._create_histogram")
@patch("src.classify3D.distributions.Distributions._create_kde_plot")
def test_get_distributions(
    mock_kde_plot: MagicMock,
    mock_histogram: MagicMock,
    mock_bar_plot: MagicMock,
    distributions: Distributions,
    mock_data: np.ndarray,
) -> None:
    """
    Test that _get_distributions returns the correct plots.

    Name
    ----
    test_get_distributions

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.
        numpy
            A library for efficient numerical computation.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_kde_plot : MagicMock
        Mock of the stats.gaussian_kde function.
    mock_histogram : MagicMock
        Mock of the px.histogram function.
    mock_bar_plot : MagicMock
        Mock of the px.bar function.
    distributions : Distributions
        Mock of the Distributions class.
    mock_data : np.ndarray
        Mock dataset.
    """
    distributions._get_distributions(y=mock_data)
    mock_bar_plot.assert_called_once()
    mock_histogram.assert_called_once()
    mock_kde_plot.assert_called_once()


@patch("src.classify3D.distributions.go.FigureWidget")
@patch("ipywidgets.VBox")
def test_call_notebook(
    mock_vbox: MagicMock,
    mock_figure_widget: MagicMock,
    distributions: Distributions,
) -> None:
    """
    Test that the __call__ method works with 'notebook' display.

    Name
    ----
    test_call_notebook

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_vbox : MagicMock
        Mock of the ipywidgets.VBox function.
    mock_figure_widget : MagicMock
        Mock of the go.FigureWidget function.
    distributions : Distributions
        Mock of the Distributions class.
    """
    distributions(display="notebook")
    mock_figure_widget.assert_called()
    mock_vbox.assert_called_once()


@patch("dash.Dash.run_server")
@patch("dash_bootstrap_components.Container")
def test_call_container(
    mock_container: MagicMock,
    mock_run_server: MagicMock,
    distributions: Distributions,
) -> None:
    """
    Test that the __call__ method works with 'container' display.

    Name
    ----
    test_call_container

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        distributions
            Module that contains the Distributions class.

    Parameters
    ----------
    mock_container : MagicMock
        Mock of the dash_bootstrap_components.Container function.
    mock_run_server : MagicMock
        Mock of the Dash.run_server function.
    distributions : Distributions
        Mock of the Distributions class.
    """
    distributions(display="container")
    mock_container.assert_called_once()
    mock_run_server.assert_called_once()
