"""
Test the ClassStatistics class.

Name
----
test_class_statistics.py

Dependencies
------------
    External
    --------
    pytest
        Package for creating fixtures.
    numpy
        Package for working with arrays.
    dash
        Package for creating web applications.
    dash_bootstrap_components
        Package for creating Bootstrap components.

    Internal
    --------
    class_statistics
        Module containing the ClassStatistics class.

Attributes
----------
    Functions
    ---------
    test_data()
        Fixture for reusable data arrays.
    class_stats()
        Fixture for ClassStatistics instance.
    test_repr()
        Test the __repr__ method.
    test_split_data()
        Test the _split_data method.
    test_ligthten_color()
        Test the _lighten_color method.
    test_create_layout()
        Test the layout creation (Mocking Dash components).
    test_register_callbacks()
        Test Dash callback registration (Mocking the Dash app).
    test_display_page()
        Test _display_page method.
    test_call_container()
        Mock Dash app and run_server for testing __call__.
    test_update_page()
        Test _update_page method.
    test_update_dropdown_style()
        Test the _update_dropdown_style method.
    test_update_button_styles()
        Test the _update_button_styles method.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dash import Dash, html
import dash_bootstrap_components as dbc
from src.classify3D.basic_statistics import BasicStatistics
from src.classify3D.class_statistics import ClassStatistics


@pytest.fixture
def test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Fixture for reusable data arrays.

    Name
    ----
    test_data

    Dependencies
    ------------
        External
        --------
        numpy
            Library for working with arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the test data.
    """
    X = np.array(
        object=[
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11, 12],
        ]
    )
    y = np.array(object=['#000000', '#111111', '#000000', '#111111'])
    return X, y


@pytest.fixture
def class_stats(test_data: tuple[np.ndarray, np.ndarray]) -> ClassStatistics:
    """
    Fixture for ClassStatistics instance.

    Name
    ----
    class_stats

    Dependencies
    ------------
        External
        --------
        numpy
            Library for working with arrays.

        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    test_data : tuple[np.ndarray, np.ndarray]
        A tuple containing the test data.

    Returns
    -------
    ClassStatistics
        An instance of the ClassStatistics class.
    """
    X, y = test_data
    return ClassStatistics(X=X, y=y)


def test_repr(class_stats: ClassStatistics) -> None:
    """
    Test the __repr__ method.

    Name
    ----
    test_repr

    Dependencies
    ------------
        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    assert repr(class_stats) == "ClassStatistics(number_of_classes=2)"


def test_split_data(
    class_stats: ClassStatistics, test_data: tuple[np.ndarray, np.ndarray]
) -> None:
    """
    Test the _split_data method.

    Name
    ----
    test_split_data

    Dependencies
    ------------
        External
        --------
        numpy
            Library for working with arrays.

        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    test_data : tuple[np.ndarray, np.ndarray]
        A tuple containing the test data.
    """
    X, y = test_data
    split_data = class_stats._split_data(X=X, y=y)

    assert len(split_data) == 2
    assert '#000000' in split_data
    assert '#111111' in split_data
    assert isinstance(split_data['#000000'], BasicStatistics)
    assert isinstance(split_data['#111111'], BasicStatistics)


def test_lighten_color(class_stats: ClassStatistics) -> None:
    """
    Test the _lighten_color method.

    Name
    ----
    test_lighten_color

    Dependencies
    ------------
        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    color = "#FF0000"  # Bright red
    lightened_color = class_stats._lighten_color(color=color, amount=0.5)

    # Check that the color is a valid hex and different from the original
    assert lightened_color.startswith("#")
    assert len(lightened_color) == 7
    assert lightened_color != color


def test_create_layout(class_stats: ClassStatistics) -> None:
    """
    Test the layout creation (Mocking Dash components).

    Name
    ----
    test_create_layout

    Dependencies
    ------------
        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    layout = class_stats._create_layout()

    # Assert that the layout is a Dash Div component
    assert isinstance(layout, html.Div)


@patch.object(target=Dash, attribute='callback')
def test_register_callbacks(
    mock_callback: MagicMock, class_stats: ClassStatistics
) -> None:
    """
    Test Dash callback registration (Mocking the Dash app).

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
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    mock_callback : MagicMock
        Mock object for testing.
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    app = Dash(name=__name__)
    class_stats._register_callbacks(app=app)

    # Assert that the callback function is called at least once
    assert mock_callback.called


def test_display_page(class_stats: ClassStatistics) -> None:
    """
    Test _display_page method.

    Name
    ----
    test_display_page

    Dependencies
    ------------
        External
        --------
        dash_bootstrap_components
            Package for creating Bootstrap components.
        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    page_layout = class_stats._display_page(
        current_page_key='#000000', selected_key='#000000'
    )

    # Check if the layout is a Container
    assert isinstance(page_layout, dbc.Container)


@patch('dash.Dash.run_server')
def test_call_container(
    mock_run_server: MagicMock, class_stats: ClassStatistics
) -> None:
    """
    Mock Dash app and run_server for testing __call__.

    Name
    ----
    test_call_container

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    callback : MagicMock
        Mock object for testing.
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    # Call with 'container' display to trigger the Dash app start
    class_stats(display='container')

    # Assert that the run_server method is called
    mock_run_server.assert_called_once_with(
        debug=True, host='0.0.0.0', port=8050
    )


@patch("dash.callback_context")
def test_update_page(
    callback: MagicMock, class_stats: ClassStatistics
) -> None:
    """
    Test _update_page method.

    Name
    ----
    test_update_page

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    callback : MagicMock
        Mock object for testing.
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    # Mock the callback context object
    callback.triggered = [{"prop_id": "prev-page-class-stats.n_clicks"}]
    current_page, dropdown_value = class_stats._update_page(
        prev_clicks=1, next_clicks=0, selected_key='#000000'
    )

    # Since previous clicks are triggered, we expect the page to change
    assert current_page == '#000000'
    assert dropdown_value == '#000000'


def test_update_dropdown_style(class_stats: ClassStatistics) -> None:
    """
    Test _update_dropdown_style method.

    Name
    ----
    test_update_dropdown_style

    Dependencies
    ------------
        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    style = class_stats._update_dropdown_style(selected_key='#000000')

    # Check if the style dict contains expected keys
    assert 'color' in style
    assert 'backgroundColor' in style


def test_update_button_styles(class_stats: ClassStatistics) -> None:
    """
    Test _update_button_styles method.

    Name
    ----
    test_update_button_styles

    Dependencies
    ------------
        Internal
        --------
        class_statistics
            Module containing the ClassStatistics class.

    Parameters
    ----------
    class_stats : ClassStatistics
        An instance of the ClassStatistics class.
    """
    prev_style, next_style = class_stats._update_button_styles(
        selected_key='#000000'
    )

    # Assert that both styles are dictionaries with expected properties
    assert isinstance(prev_style, dict)
    assert isinstance(next_style, dict)
    assert 'backgroundColor' in prev_style
    assert 'backgroundColor' in next_style
