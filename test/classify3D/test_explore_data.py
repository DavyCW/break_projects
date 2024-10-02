"""
A test suite for the ExploreData class.

Name
----
test_explore_data.py

Dependencies
------------
    External
    --------
    pytest
        Package for creating fixtures.
    dash
        Package for creating web applications.
    dash_bootstrap_components
        Package for creating Bootstrap components.

    Internal
    --------
    split_data
        Module containing the SplitData class.
    explore_data
        Module containing the ExploreData class.

Attributes
----------
    Functions
    ---------
    mock_split_data()
        Fixture to mock the split data used in ExploreData.
    mock_explore_data()
        Fixture to instantiate the ExploreData class with mocked split data.
    mock_dash_app()
        Fixture to mock the Dash app.
    test_explore_data_init()
        Test the initialization of the ExploreData class.
    test_register_callbacks()
        Test that the callback registration method registers the callback.
    test_setup_layout()
        Test if the layout is properly set up in the Dash app.
    test_render_content()
        Test the content rendering based on the selected tab.
    test_explore_data_call()
        Test the ExploreData class's __call__ method.
"""

import pytest
from unittest.mock import MagicMock, patch
from dash import Dash
from src.classify3D import split_data
from src.classify3D.explore_data import ExploreData  # type: ignore
import dash_bootstrap_components as dbc


@pytest.fixture
def mock_split_data() -> MagicMock:
    """
    Fixture to mock the split data used in ExploreData.

    Name
    ----
    mock_split_data

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        split_data
            Module containing the SplitData class.

    Returns
    -------
    MagicMock
        Mock object for testing.
    """
    mock_split = MagicMock(spec=split_data.SplitData)
    mock_split.X = MagicMock()
    mock_split.y = MagicMock()
    # Mock X and y train/test/val
    mock_split.X.train = MagicMock()
    mock_split.X.test = MagicMock()
    mock_split.X.val = MagicMock()
    mock_split.y.train = MagicMock()
    mock_split.y.test = MagicMock()
    mock_split.y.val = MagicMock()

    # Mock the generator for data
    mock_split.gen = MagicMock(return_value="Mock Generator")

    return mock_split


@pytest.fixture
@patch('src.classify3D.basic_statistics.BasicStatistics')
@patch('src.classify3D.class_statistics.ClassStatistics')
@patch('src.classify3D.basic_correlation.BasicCorrelation')
@patch('src.classify3D.class_correlation.ClassCorrelation')
@patch('src.classify3D.pairplot.PairPlot')
@patch('src.classify3D.distributions.Distributions')
@patch('src.classify3D.outliers.Outliers')
@patch('src.classify3D.novelty.Novelty')
def mock_explore_data(
    mock_novelty: MagicMock,
    mock_outliers: MagicMock,
    mock_distributions: MagicMock,
    mock_pairplot: MagicMock,
    mock_class_correlation: MagicMock,
    mock_basic_correlation: MagicMock,
    mock_class_statistics: MagicMock,
    mock_basic_statistics: MagicMock,
    mock_split_data: MagicMock,
) -> ExploreData:
    """
    Fixture to instantiate the ExploreData class with mocked split data.

    Name
    ----
    mock_explore_data

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        explore_data
            Module containing the ExploreData class.

    Parameters
    ----------
    mock_novelty : MagicMock
        Mock object for testing.
    mock_outliers : MagicMock
        Mock object for testing.
    mock_distributions : MagicMock
        Mock object for testing.
    mock_pairplot : MagicMock
        Mock object for testing.
    mock_class_correlation : MagicMock
        Mock object for testing.
    mock_basic_correlation : MagicMock
        Mock object for testing.
    mock_class_statistics : MagicMock
        Mock object for testing.
    mock_basic_statistics : MagicMock
        Mock object for testing.
    mock_split_data : MagicMock
        Mock object for testing.

    Returns
    -------
    ExploreData
        An instance of the ExploreData class.
    """
    return ExploreData(split=mock_split_data)


@pytest.fixture
def mock_dash_app() -> Dash:
    """
    Fixture to create a Dash app instance.

    Name
    ----
    mock_dash_app

    Dependencies
    ------------
        External
        --------
        dash
            Package for creating reactive web applications.

    Returns
    -------
    Dash
        An instance of the Dash class.
    """
    return Dash(name=__name__)


@patch('src.classify3D.basic_statistics.BasicStatistics')
@patch('src.classify3D.class_statistics.ClassStatistics')
@patch('src.classify3D.basic_correlation.BasicCorrelation')
@patch('src.classify3D.class_correlation.ClassCorrelation')
@patch('src.classify3D.pairplot.PairPlot')
@patch('src.classify3D.distributions.Distributions')
@patch('src.classify3D.outliers.Outliers')
@patch('src.classify3D.novelty.Novelty')
def test_explore_data_init(
    mock_novelty: MagicMock,
    mock_outliers: MagicMock,
    mock_distributions: MagicMock,
    mock_pairplot: MagicMock,
    mock_class_correlation: MagicMock,
    mock_basic_correlation: MagicMock,
    mock_class_statistics: MagicMock,
    mock_basic_statistics: MagicMock,
    mock_split_data: MagicMock,
) -> None:
    """
    Test the initialization of the ExploreData class.

    Name
    ----
    test_explore_data_init

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        explore_data
            Module containing the ExploreData class.

    Parameters
    ----------
    mock_novelty : MagicMock
        Mock object for testing.
    mock_outliers : MagicMock
        Mock object for testing.
    mock_distributions : MagicMock
        Mock object for testing.
    mock_pairplot : MagicMock
        Mock object for testing.
    mock_class_correlation : MagicMock
        Mock object for testing.
    mock_basic_correlation : MagicMock
        Mock object for testing.
    mock_class_statistics : MagicMock
        Mock object for testing.
    mock_basic_statistics : MagicMock
        Mock object for testing.
    mock_split_data : MagicMock
        Mock object for testing.
    """
    # Initialize ExploreData
    explore_data = ExploreData(mock_split_data)

    # Check if the objects were created properly
    mock_basic_statistics.assert_called()
    mock_class_statistics.assert_called()
    mock_basic_correlation.assert_called()
    mock_class_correlation.assert_called()
    mock_pairplot.assert_called()
    mock_distributions.assert_called()
    mock_outliers.assert_called()
    mock_novelty.assert_called()

    assert explore_data.gen == mock_split_data.gen
    assert isinstance(explore_data.basic_stats_train, MagicMock)
    assert isinstance(explore_data.basic_stats_test, MagicMock)
    assert isinstance(explore_data.basic_stats_val, MagicMock)


def test_register_callbacks(
    mock_explore_data: ExploreData, mock_dash_app: Dash
) -> None:
    """
    Test that the callback registration method registers the callback.

    Name
    ----
    test_register_callbacks

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.
        dash
            Package for creating web applications.

        Internal
        --------
        explore_data
            Module containing the ExploreData class.

    Parameters
    ----------
    mock_explore_data : ExploreData
        An instance of the ExploreData class.
    mock_dash_app : Dash
        An instance of the Dash class.
    """
    mock_layouts = {'gen': MagicMock(), 'basic_stats_train': MagicMock()}

    # Register callbacks
    mock_explore_data._register_callbacks(mock_dash_app, mock_layouts)

    # Check if the callback was registered
    assert 'tabs-content.children' in mock_dash_app.callback_map


def test_setup_layout(
    mock_explore_data: ExploreData, mock_dash_app: Dash
) -> None:
    """
    Test if the layout is properly set up in the Dash app.

    Name
    ----
    test_setup_layout

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.
        dash
            Package for creating web applications.
        dash_bootstrap_components
            Package for creating Bootstrap components.

        Internal
        --------
        explore_data
            Module containing the ExploreData class.

    Parameters
    ----------
    mock_explore_data : ExploreData
        An instance of the ExploreData class.
    mock_dash_app : Dash
        An instance of the Dash class.
    """
    mock_explore_data._setup_layout(mock_dash_app)

    # Check if the layout has been assigned
    assert mock_dash_app.layout is not None
    assert isinstance(mock_dash_app.layout, dbc.Container)


def test_render_content(mock_explore_data: ExploreData) -> None:
    """
    Test the content rendering based on the selected tab.

    Name
    ----
    test_render_content

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        explore_data
            Module containing the ExploreData class.

    Parameters
    ----------
    mock_explore_data : ExploreData
        An instance of the ExploreData class.
    """
    mock_layouts = {
        'gen': 'Generator Layout',
        'basic_stats_train': 'Basic Stats Train Layout',
    }

    # Check rendering for 'gen' tab
    content = mock_explore_data._render_content(mock_layouts, 'gen')
    assert content == 'Generator Layout'

    # Check rendering for 'basic_stats_train' tab
    content = mock_explore_data._render_content(
        mock_layouts, 'basic_stats_train'
    )
    assert content == 'Basic Stats Train Layout'

    # Check for non-existent tab
    content = mock_explore_data._render_content(mock_layouts, 'nonexistent')
    assert content == "Tab not found"


@patch('dash.Dash.run_server')
def test_explore_data_call(
    mock_run_server: MagicMock, mock_explore_data: ExploreData
) -> None:
    """
    Test the ExploreData class's __call__ method.

    Name
    ----
    test_explore_data_call

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        explore_data
            Module containing the ExploreData class.

    Parameters
    ----------
    mock_run_server : MagicMock
        Mock object for testing.
    mock_explore_data : ExploreData
        An instance of the ExploreData class.
    """
    mock_explore_data()
    # Assert that the run_server method is called
    mock_run_server.assert_called_once_with(
        debug=True, host='0.0.0.0', port=8050
    )
