"""
A test suite for the PairPlot class.

Name
----
test_pairplot.py

Dependencies
------------
    External
    --------
    numpy
        Package for fast array manipulation.
    pytest
        A library for writing and running tests.
    plotly
        Package to plot 3D data beautifully.
    dash_bootstrap_components
        Package for creating Bootstrap components.
    ipywidgets
        Package for making interactive web applications.

    Internal
    --------
    pairplot
        Module containing the PairPlot class.

Attributes
----------
    Functions
    ---------
    sample_X()
        Fixture for a sample feature matrix (X).
    sample_y()
        Fixture for a sample label array (y).
    pairplot_instance()
        Fixture for creating a PairPlot instance.
    test_pairplot_init()
        Test the initialization of the PairPlot class.
    test_pairplot_repr()
        Test the __repr__ method.
    test_get_pairplot()
        Test if _get_pairplot returns a valid Figure object.
    test_pairplot_call()
        Test the __call__ method for 'container' display.
    test_pairplot_invalid_display()
        Test that an invalid display option raises a ValueError.
    test_pairplot_notebook()
        Test the __call__ method for 'notebook' display.
    test_pairplot_none()
        Test the __call__ method for 'none' display.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import dash_bootstrap_components as dbc
from plotly.graph_objects import Figure
from src.classify3D.pairplot import PairPlot  # type: ignore
import ipywidgets as widgets


@pytest.fixture
def sample_X() -> np.ndarray:
    """
    Fixture for a sample feature matrix (X).

    Name
    ----
    sample_X

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

    Returns
    -------
    np.ndarray
        Sample feature matrix.
    """
    return np.random.rand(3000, 6)


@pytest.fixture
def sample_y() -> np.ndarray:
    """
    Fixture for a sample label array (y).

    Name
    ----
    sample_y

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

    Returns
    -------
    np.ndarray
        Sample label data.
    """
    return np.random.randint(0, 2, size=3000)


@pytest.fixture
def pairplot_instance(sample_X: np.ndarray, sample_y: np.ndarray) -> PairPlot:
    """
    Fixture for creating a PairPlot instance.

    Name
    ----
    pairplot_instance

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    sample_X : np.ndarray
        Sample feature data.
    sample_y : np.ndarray
        Sample label data.

    Returns
    -------
    PairPlot
        Instance of the PairPlot class.
    """
    with patch('numpy.random.choice', return_value=np.arange(2560)):
        return PairPlot(X=sample_X, y=sample_y, sample_size=2560)


def test_pairplot_init(pairplot_instance: PairPlot) -> None:
    """
    Test the initialization of the PairPlot class.

    Name
    ----
    test_pairplot_init

    Dependencies
    ------------
        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    """
    assert isinstance(pairplot_instance, PairPlot)
    assert pairplot_instance.sample_size == 2560


def test_pairplot_repr(pairplot_instance: PairPlot) -> None:
    """
    Test the __repr__ method.

    Name
    ----
    test_pairplot_repr

    Dependencies
    ------------
        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    """
    assert repr(pairplot_instance) == "PairPlot()"


def test_get_pairplot(
    pairplot_instance: PairPlot, sample_X: np.ndarray, sample_y: np.ndarray
) -> None:
    """
    Test if _get_pairplot returns a valid Figure object.

    Name
    ----
    test_get_pairplot

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.
        plotly
            Package to plot 3D data beautifully.

        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    sample_X : np.ndarray
        Sample feature matrix.
    sample_y : np.ndarray
        Sample label array.
    """
    fig = pairplot_instance._get_pairplot(X=sample_X, y=sample_y)
    assert isinstance(fig, Figure)
    assert len(fig.data) > 0  # There should be data in the plot


@patch("src.classify3D.pairplot.dash.Dash.run_server")
def test_pairplot_call(
    mock_run_server: MagicMock, pairplot_instance: PairPlot
) -> None:
    """
    Test the __call__ method for 'container' display.

    Name
    ----
    test_pairplot_call

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    mock_run_server : MagicMock
        _description_
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    """
    # Test the "container" option which starts a Dash server
    pairplot_instance(display='container')
    mock_run_server.assert_called_once_with(
        debug=True, host='0.0.0.0', port=8050
    )


def test_pairplot_invalid_display(pairplot_instance: PairPlot) -> None:
    """
    Test that an invalid display option raises a ValueError.

    Name
    ----
    test_pairplot_invalid_display

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    """
    with pytest.raises(
        expected_exception=ValueError,
        match="Invalid display option 'invalid'.*",
    ):
        pairplot_instance(display='invalid')


def test_pairplot_notebook(pairplot_instance: PairPlot) -> None:
    """
    Test the __call__ method for 'notebook' display.

    Name
    ----
    test_pairplot_notebook

    Dependencies
    ------------
        External
        --------
        ipywidgets
            Package for making interactive web applications.

        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    """
    result = pairplot_instance(display='notebook')
    assert isinstance(result, widgets.VBox)  # Should return a VBox container


def test_pairplot_none(pairplot_instance: PairPlot) -> None:
    """
    Test the __call__ method for 'none' display.

    Name
    ----
    test_pairplot_none

    Dependencies
    ------------
        External
        --------
        dash_bootstrap_components
            Package for creating Bootstrap components.

        Internal
        --------
        pairplot
            Module containing the PairPlot class.

    Parameters
    ----------
    pairplot_instance : PairPlot
        Instance of the PairPlot class.
    """
    result = pairplot_instance(display='none')
    assert isinstance(result, dbc.Container)  # Should return a Dash app layout
