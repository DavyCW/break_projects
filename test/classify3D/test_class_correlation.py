"""
Module for testing ClassCorrelation class and its methods.

Name
----
test_class_correlation.py

Description
-----------
Module for testing ClassCorrelation class and its methods.

Dependencies
------------
    External
    --------
    pytest
        Package for creating fixtures.
    dash
        Package for creating web applications.
    numpy
        Package for scientific computing.

    Internal
    --------
    class_correlation
        A class representing class correlation of a dataset.

Attributes
----------
    Functions
    ---------
    class_cor_obj()
        Fixture for ClassCorrelation class.
    test_init_and_repr()
        Test the constructor and __repr__ method.
    test_split_data()
        Test the _split_data method for proper splitting and key creation.
    test_lighten_color()
        Test the _lighten_color function for proper color transformation.
    test_create_layout()
        Test the _create_layout method for proper layout creation.
    test_display_page()
        Test the _display_page function for correct page retrieval.
    test_update_page()
        Test the _update_page function for correct page navigation.
    test_update_dropdown_style()
        Test the _update_dropdown_style for proper color updates.
    test_update_button_styles()
        Test the _update_button_styles function for proper button styling.
    test_notebook_display()
        Test the notebook display method (mocking notebook display).
"""

import numpy as np
from unittest.mock import MagicMock, patch
import pytest
from dash import html, dcc
from src.classify3D.class_correlation import ClassCorrelation


@pytest.fixture
@patch("src.classify3D.class_correlation.BasicCorrelation", autospec=True)
def class_cor_obj(mock_basic: MagicMock) -> tuple[ClassCorrelation, MagicMock]:
    """
    Fixture for ClassCorrelation class.

    Name
    ----
    class_cor_obj

    Description
    -----------
    Fixture for ClassCorrelation class.

    Dependencies
    ------------
        External
        --------
        numpy
            Package for scientific computing.
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    mock_basic : MagicMock
        Mock object for BasicCorrelation

    Returns
    -------
    tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    mock_page = MagicMock()
    mock_basic.return_value = mock_page
    return (
        ClassCorrelation(
            X=np.random.rand(5, 6),
            y=np.array(
                object=['#000000', '#000000', '#111111', '#111111', '#222222']
            ),
        ),
        mock_page,
    )


def test_init_and_repr(
    class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the constructor and __repr__ method.

    Name
    ----
    test_init_and_repr

    Description
    -----------
    Test the constructor and __repr__ method.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    # Check if the dictionary was created with the correct keys (unique labels)
    assert set(class_cor_obj[0].keys) == {'#000000', '#111111', '#222222'}
    assert repr(class_cor_obj[0]) == "ClassCorrelation(number_of_classes=3)"


def test_split_data(class_cor_obj: tuple[ClassCorrelation, MagicMock]) -> None:
    """
    Test the _split_data method for proper splitting and key creation.

    Name
    ----
    test_split_data

    Description
    -----------
    Test the _split_data method for proper splitting and key creation.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    # Check if split_data correctly groups data and assigns BasicCorrelation
    assert set(class_cor_obj[0].basic_correlation_dict.keys()) == {
        '#000000',
        '#111111',
        '#222222',
    }
    class_cor_obj[1].call_count == 3


def test_lighten_color(
    class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the _lighten_color function for proper color transformation.

    Name
    ----
    test_lighten_color

    Description
    -----------
    Test the _lighten_color function for proper color transformation.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    color = "#ff0000"  # Red
    result = class_cor_obj[0]._lighten_color(color=color, amount=0.5)

    # Verify the output is a valid hex color and is lighter than the input
    assert isinstance(result, str)
    assert result.startswith('#')


def test_create_layout(
    class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the _create_layout method for proper layout creation.

    Name
    ----
    test_create_layout

    Description
    -----------
    Test the _create_layout method for proper layout creation.

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
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    layout = class_cor_obj[0]._create_layout()

    # Check if layout is an instance of html.Div and contains certain elements
    assert isinstance(layout, html.Div)
    assert isinstance(layout.children[0], dcc.Location)


def test_display_page(
    class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the _display_page function for correct page retrieval.

    Name
    ----
    test_display_page

    Description
    -----------
    Test the _display_page function for correct page retrieval.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    result = class_cor_obj[0]._display_page(
        current_page_key='#000000', selected_key='#222222'
    )

    # Ensure the function returns the correct page's content
    assert result == class_cor_obj[1]('none')


@patch("dash.callback_context")
def test_update_page(
    mock_callback_context: MagicMock,
    class_cor_obj: tuple[ClassCorrelation, MagicMock],
) -> None:
    """
    Test the _update_page function for correct page navigation.

    Name
    ----
    test_update_page

    Description
    -----------
    Test the _update_page function for correct page navigation.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    mock_callback_context : MagicMock
        Mocked callback context object.
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    # Mock the callback context object
    mock_callback_context.triggered = [
        {"prop_id": "next-page-class-cor.n_clicks"}
    ]
    # Simulate clicking "next" from page "0"
    current_page, current_dropdown = class_cor_obj[0]._update_page(
        prev_clicks=0, next_clicks=1, selected_key='#000000'
    )
    assert current_page == '#111111'
    assert current_dropdown == '#111111'

    # Simulate clicking "prev" from page "#111111"
    mock_callback_context.triggered = [
        {"prop_id": "prev-page-class-cor.n_clicks"}
    ]

    current_page, current_dropdown = class_cor_obj[0]._update_page(
        prev_clicks=1, next_clicks=0, selected_key='#111111'
    )
    assert current_page == '#000000'
    assert current_dropdown == '#000000'


def test_update_dropdown_style(
    class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the _update_dropdown_style for proper color updates.

    Name
    ----
    test_update_dropdown_style

    Description
    -----------
    Test the _update_dropdown_style for proper color updates.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    style = class_cor_obj[0]._update_dropdown_style(selected_key='#000000')

    # Ensure that the background color is updated correctly
    assert 'backgroundColor' in style


def test_update_button_styles(
    class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the _update_button_styles function for proper button styling.

    Name
    ----
    test_update_button_styles

    Description
    -----------
    Test the _update_button_styles function for proper button styling.

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    prev_style, next_style = class_cor_obj[0]._update_button_styles(
        selected_key='#000000'
    )

    # Ensure the buttons have the correct color styles
    assert 'backgroundColor' in prev_style
    assert 'backgroundColor' in next_style


@patch("src.classify3D.class_correlation.notebook_display")
def test_notebook_display(
    mock_display: MagicMock, class_cor_obj: tuple[ClassCorrelation, MagicMock]
) -> None:
    """
    Test the notebook display method (mocking notebook display).

    Name
    ----
    test_notebook_display

    Description
    -----------
    Test the notebook display method (mocking notebook display).

    Dependencies
    ------------
        External
        --------
        pytest
            Package for creating fixtures.

        Internal
        --------
        class_correlation
            A class representing class correlation of a dataset.

    Parameters
    ----------
    mock_display : MagicMock
        Mock the notebook display function.
    class_cor_obj : tuple[ClassCorrelation, MagicMock]
        Fixture for ClassCorrelation class.
    """
    # Call the method and check if the display was triggered
    class_cor_obj[0]._notebook_display()
    mock_display.assert_called()
