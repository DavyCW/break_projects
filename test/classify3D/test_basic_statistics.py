"""
Module for testing BasicStatistics class and its methods.

Name
----
test_basic_statistics.py

Description
-----------
This module contains functions and classes for testing the basic_statistics
module.

Dependencies
------------
    External
    --------
    numpy
        A library for efficient numerical computation.
    pandas
        Package for data manipulation and analysis.
    pytest
        A library for writing and running tests.

    Internal
    --------
    basic_statistics
        Module calculates and displays basic statistics of a dataset.

Attributes
----------
    Functions
    ---------
    test_get_basic_statistics_dataframe()
        Test that the basic statistics dataframe is calculated correctly.
    test_lighten_color()
        Test the _lighten_color method to ensure it lightens the color.
    test_dash_container_display()
        Test that the Dash server starts when 'container' display is chosen.
    test_notebook_display()
        Test that the statistics are displayed in a Jupyter notebook.
    test_invalid_display_option()
        Test that an invalid display option raises a ValueError.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.classify3D.basic_statistics import BasicStatistics


def test_get_basic_statistics_dataframe() -> None:
    """
    Test that the basic statistics dataframe is calculated correctly.

    Name
    ----
    test_get_basic_statistics_dataframe

    Description
    -----------
    Test that the basic statistics dataframe is calculated correctly.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pandas
            Package for data manipulation and analysis.

        Internal
        --------
        basic_statistics
            Module calculates and displays basic statistics of a dataset.
    """
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )  # Sample data
    expected_df = pd.DataFrame(
        data={
            'mean': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'median': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'std': [2.44949, 2.44949, 2.44949, 2.44949, 2.44949, 2.44949],
            'minimum': [1, 2, 3, 4, 5, 6],
            'maximum': [7, 8, 9, 10, 11, 12],
        },
        index=['x', 'y', 'z', 'r', 'theta', 'phi'],
    )

    stats = BasicStatistics(X=X, name="#00b5eb")
    pd.testing.assert_frame_equal(
        left=stats.basic_statistics, right=expected_df
    )


def test_lighten_color() -> None:
    """
    Test the _lighten_color method to ensure it lightens the color.

    Name
    ----
    test_lighten_color

    Description
    -----------
    Test the _lighten_color method to ensure it lightens the color.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.

        Internal
        --------
        basic_statistics
            Module calculates and displays basic statistics of a dataset.
    """
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )  # Sample data
    stats = BasicStatistics(X=X, name='#FF0000')
    lightened_color = stats._lighten_color(color='#FF0000', amount=0.3)

    assert lightened_color == '#ffb2b2', "Expected a lighter red color"


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
        basic_statistics
            Module calculates and displays basic statistics of a dataset.

    Parameters
    ----------
    mock_run_server : MagicMock
        Mock run_server object
    """
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )  # Sample data
    stats = BasicStatistics(X=X, name="#00b5eb")

    stats('container')

    # Ensure Dash's run_server was called, which implies use of the container.
    mock_run_server.assert_called_once()


@patch('src.classify3D.basic_statistics.notebook_display')
def test_notebook_display(mock_display: MagicMock) -> None:
    """
    Test that the statistics are displayed in a Jupyter notebook.

    Name
    ----
    test_notebook_display

    Description
    -----------
    Test that the statistics are displayed in a Jupyter notebook.

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
        basic_statistics
            Module calculates and displays basic statistics of a dataset.

    Parameters
    ----------
    mock_display : MagicMock
        Mock display object
    """
    # Sample data
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )

    # Create the BasicStatistics object
    stats = BasicStatistics(X=X, name="#00b5eb")

    # Check for null pointer references
    assert stats is not None, "BasicStatistics object should not be null"

    # Call the method to display in the notebook
    stats('notebook')

    # Ensure the display was called once
    mock_display.assert_called_once()


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
        basic_statistics
            Module calculates and displays basic statistics of a dataset.

    Parameters
    ----------
    invalid_display : str
        The invalid display option.
    """
    X = np.array(
        object=[[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12]]
    )
    stats = BasicStatistics(X=X, name="#00b5eb")

    with pytest.raises(expected_exception=ValueError):
        stats(invalid_display)
