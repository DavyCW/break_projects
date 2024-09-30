"""
Module for testing BasicCorrelation class and its methods.

Name
----
test_basic_correlation.py

Description
-----------
This module contains functions and classes for testing the basic_correlation
module.

Dependencies
------------
    External
    --------
    numpy
        A library for efficient numerical computation.
    pytest
        A library for writing and running tests.
    ipywidgets
        A library for creating interactive user interfaces.

    Internal
    --------
    basic_correlation
        Module calculates and displays basic correlation of a dataset.

Attributes
----------
    Functions
    ---------
    test_dash_container_display()
        Test that the Dash server starts when 'container' display is chosen.
    test_notebook_display()
        Test that the correlations are displayed in a Jupyter notebook.
    test_invalid_display_option()
        Test that an invalid display option raises a ValueError.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.classify3D.basic_correlation import BasicCorrelation
import ipywidgets as widgets


@patch('dash.Dash.run_server')
def test_dash_container_display(mock_run_server: MagicMock) -> None:
    """
    Test that the Dash server starts when 'container' display is chosen.

    Name
    ----
    test_dash_container_display

    Description
    -----------
    Test that the Dash server starts when 'container' display is chosen.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pytest
            A library for writing and running tests.

        Internal
        --------
        basic_correlation
            Module calculates and displays basic correlation of a dataset.

    Parameters
    ----------
    mock_run_server : MagicMock
        Mock run_server object
    """
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )  # Sample data
    cor = BasicCorrelation(X=X, name="#00b5eb")

    cor('container')

    # Ensure Dash's run_server was called, which implies use of the container.
    mock_run_server.assert_called_once()


def test_notebook_display() -> None:
    """
    Test that the correlations are displayed in a Jupyter notebook.

    Name
    ----
    test_notebook_display

    Description
    -----------
    Test that the correlations are displayed in a Jupyter notebook.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        ipywidgets
            A library for creating interactive user interfaces.

        Internal
        --------
        basic_correlation
            Module calculates and displays basic correlation of a dataset.
    """
    # Sample data
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )

    # Create the BasicStatistics object
    cor = BasicCorrelation(X=X, name="#00b5eb")

    # Check for null pointer references
    assert cor is not None, "BasicCorrelation object should not be null"

    # Call the method to display in the notebook
    container = cor('notebook')

    # Ensure the display was called once
    assert isinstance(container, widgets.VBox)


@pytest.mark.parametrize("invalid_display", ["invalid_mode", "wrong_display"])
def test_invalid_display_option(invalid_display: str) -> None:
    """
    Test that an invalid display option raises a ValueError.

    Name
    ----
    test_invalid_display_option

    Description
    -----------
    Test that an invalid display option raises a ValueError.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pytest
            A library for writing and running tests.

        Internal
        --------
        basic_correlation
            Module calculates and displays basic correlation of a dataset.

    Parameters
    ----------
    invalid_display : str
        The invalid display option.
    """
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )
    cor = BasicCorrelation(X=X, name="#00b5eb")

    with pytest.raises(expected_exception=ValueError):
        cor(invalid_display)
