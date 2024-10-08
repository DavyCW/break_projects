"""
Module for displaying basic correlation of a dataset split by class.

Name
----
class_correlation.py

Description
-----------
This module provides a class for displaying basic correlation of a dataset
split by class.

Dependencies
------------
    External
    --------
    numpy
        Package for fast array manipulation.
    dash
        Package for making reactive web applications.
    matplotlib
        Package for good color distributions.
    ipykernel
        Package for running notebooks in a web browser.
    ipywidgets
        Package for making interactive web applications.
    dash_bootstrap_components
            Package for creating Bootstrap components.

    Internal
    --------
    BasicCorrelation
        A class representing basic correlation of a dataset.
    GenerateData
        Generates 3D data based on passed file.
    SplitData
        A class responsible for splitting the data.

Attributes
----------
    Classes
    -------
    ClassCorrelation
        A class representing basic correlation of a dataset split by class.
"""

from IPython.display import clear_output
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash
from src.classify3D.basic_correlation import BasicCorrelation  # type: ignore
from src.classify3D.generate_data import GenerateData  # type: ignore
from src.classify3D.split_data import SplitData  # type: ignore
import colorsys
from matplotlib import colors as mcolors
from IPython.display import display as notebook_display
import ipywidgets as widgets
from typing import Optional


class ClassCorrelation:
    """
    A class representing basic correlation of a dataset split by class.

    Name
    ----
    ClassCorrelation

    Description
    -----------
    This class provides methods for calculating basic correlation of a dataset
    split by class. It also provides methods for displaying the correlation in
    a dashboard.

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.
        dash
            Package for making reactive web applications.
        matplotlib
            Package for good color distributions.
        ipykernel
            Package for running notebooks in a web browser.
        ipywidgets
            Package for making interactive web applications.
        dash_bootstrap_components
            Package for creating Bootstrap components.

        Internal
        --------
        BasicCorrelation
            A class representing basic correlation of a dataset.
        GenerateData
            Generates 3D data based on passed file.
        SplitData
            A class responsible for splitting the data.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the ClassCorrelation object with a dataset.
        __repr__()
            Provide a string representation of the ClassCorrelation object.
        _split_data()
            Split the data into groups based on the unique labels.
        _lighten_color()
            Lighten the passed color by a given amount.
        _create_layout()
            Create the layout for the Dash app.
        _register_callbacks()
            Register the callbacks for the Dash app.
        _display_page()
            Display the page based on the current page key and selected key.
        _update_page()
            Update the page based on the previous and next clicks.
        _update_dropdown_style()
            Update the style of the dropdown menu based on the selected key.
        _update_button_styles()
            Update the styles of the buttons based on the selected key.
        _notebook_display()
            Display a dropdown widget that allows the user to select a class.
        _update_display()
            Update the output widget based on the selected class from the
            dropdown.
        __call__()
            Display the class correlation in a Dash app or Jupyter notebook.

        Variables
        ---------
        basic_correlation_dict : dict[str, BasicCorrelation]
            A dictionary of basic correlation for each class.
        keys : list[str]
            A list of the keys in the `basic_correlation_dict` dictionary.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the ClassCorrelation object with a dataset.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the ClassCorrelation object with a dataset.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        X : np.ndarray
            The feature data.
        y : np.ndarray
            The label data.

        Examples
        --------
        >>> class_correlation = ClassCorrelation(X=np.array([1, 2, 3, 4, 5]),
        ...                                      y=np.array([0, 0, 1, 1, 1])
        ...                                      )
        """
        self.basic_correlation_dict = self._split_data(X=X, y=y)
        self.keys = list(self.basic_correlation_dict.keys())

    def __repr__(self) -> str:
        """
        Provide a string representation of the ClassCorrelation object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the ClassCorrelation object.

        Returns
        -------
        str
            A string representation of the ClassCorrelation object.

        Examples
        --------
        >>> print(class_correlation)
        ClassCorrelation(number_of_classes=2)
        """
        return f"ClassCorrelation(number_of_classes={len(self.keys)})"

    def _split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, BasicCorrelation]:
        """
        Split the data into groups based on the unique labels.

        Name
        ----
        _split_data

        Description
        -----------
        This method takes in the feature data `X` and the label data `y` as
        parameters. It uses the `np.argsort` function to sort the data based
        on the labels, and the `np.unique` function to identify the unique
        labels and their corresponding indices. It then groups the data into
        a dictionary where the keys are the unique labels and the values are
        the corresponding data.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

            Internal
            --------
            BasicCorrelation
                A class representing basic correlation of a dataset.

        Parameters
        ----------
        X : np.ndarray
            The feature data.
        y : np.ndarray
            The label data.

        Returns
        -------
        dict[str, BasicCorrelation]
            A dictionary where the keys are the unique labels and the values
            are the corresponding data.

        Examples
        --------
        >>> class_cor = ClassCorrelation()
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 0, 1])
        >>> data_dict = class_cor._split_data(X, y)
        >>> data_dict
        {0: BasicCorrelation(data=np.array([[1, 2], [3, 4]])),
         1: BasicCorrelation(data=np.array([[5, 6]]))
         }
        """
        # Sort indices based on y
        sorted_idx = np.argsort(a=y)

        # Identify unique labels and the starting index of each group
        unique_labels, group_indices = np.unique(
            ar=y[sorted_idx], return_index=True
        )

        # Calculate the end indices for each group by shifting group_indices
        split_indices = np.append(arr=group_indices[1:], values=len(y))

        # Split X into groups based on label
        grouped_data = np.split(
            ary=X[sorted_idx], indices_or_sections=split_indices[:-1]
        )

        # Build a dictionary of data grouped by unique labels
        data_dict = {}
        for label, data in zip(unique_labels, grouped_data):
            data_dict[label] = BasicCorrelation(X=data, name=label)

        return data_dict

    def _lighten_color(self, color: str, amount=0.5) -> str:
        """
        Lighten the passed color by a given amount.

        Name
        ----
        _lighten_color

        Description
        -----------
        Turns a color from hex into rgb, then from rgb to hls, then modifies
        the hls colors to lighten, then turns back into rgb then back into hex.

        Dependencies
        ------------
            External
            --------
            matplotlib
                Package for good color distributions.

        Parameters
        ----------
        color : str
            Hex color to change.
        amount : float, optional
            The amount to change the color by, by default 0.5

        Returns
        -------
        str
            The changed hex color.

        Examples
        --------
        >>> obj = YourClassName()
        >>> obj._lighten_color('#FF0000', 0.3)
        '#FF7F00'
        """
        hls = colorsys.rgb_to_hls(*mcolors.to_rgb(c=color))
        rgb = colorsys.hls_to_rgb(
            h=hls[0], l=1 - amount * (1 - hls[1]), s=hls[2]
        )
        return mcolors.rgb2hex(c=rgb)

    def _create_layout(self) -> html.Div:
        """
        Create the layout for the Dash app.

        Name
        ----
        _create_layout

        Description
        -----------
        The layout consists of a container with a row containing buttons and
        a dropdown for navigation, and a container for the page content.

        Dependencies
        ------------
            External
            --------
            dash
                Package for creating reactive web applications.
            dash_bootstrap_components
                Package for creating Bootstrap components.

        Returns
        -------
        html.Div
            The HTML div containing the layout for the Dash app.

        Examples
        --------
        >>> class_cor = ClassCorrelation()
        >>> layout = class_cor._create_layout()
        >>> layout
        html.Div(children=[
        ...    html.H1('Class Correlation'),
        ...    dcc.Dropdown(id='class-dropdown-class-cor', options=[...]),
        ...    html.Div(id='page-content-class-cor')
        ... ])
        """
        return html.Div(
            children=[
                # Creates a Location component to get the current URL
                dcc.Location(id='url-class-cor', refresh=False),
                # Creates a container for the page content
                dbc.Container(id='page-content-class-cor', fluid=True),
                # Creates a Store component to store the current page
                dcc.Store(id='current-page-class-cor', data=self.keys[0]),
                # Creates a row for buttons and dropdown
                dbc.Row(
                    children=[
                        # Creates a column for the Previous button
                        dbc.Col(
                            children=dbc.Button(
                                children='Previous',
                                id='prev-page-class-cor',
                                n_clicks=0,
                                style={
                                    'color': '#000000',
                                    'backgroundColor': self.keys[0],
                                },
                            ),
                            width='auto',
                        ),
                        # Creates a column for the dropdown
                        dbc.Col(
                            children=dcc.Dropdown(
                                id='page-dropdown-class-cor',
                                clearable=False,
                                options=[
                                    {'label': key, 'value': key}
                                    for key in self.keys
                                ],
                                value=self.keys[0],  # Default to the first key
                                style={
                                    'width': '100%',
                                    'color': '#000000',
                                    'backgroundColor': self.keys[0],
                                },
                            ),
                            width=2,
                        ),  # Adjust width as needed
                        # Creates a column for the Next button
                        dbc.Col(
                            children=dbc.Button(
                                children='Next',
                                id='next-page-class-cor',
                                n_clicks=0,
                                style={
                                    'color': '#000000',
                                    'backgroundColor': self.keys[0],
                                },  # Button color
                            ),
                            width='auto',
                        ),
                    ],
                    justify='center',
                    align='center',
                    style={'padding-top': '20px'},
                ),
            ]
        )

    def _register_callbacks(self, app: Dash) -> None:
        """
        Register the callbacks for the Dash app.

        Name
        ----
        _register_callbacks

        Description
        -----------
        This method takes in the Dash app as a parameter. It uses the app's
        callback function to register the callbacks for the app.

        Dependencies
        ------------
            External
            --------
            dash
                Package for creating reactive web applications.

        Parameters
        ----------
        app : Dash
            The Dash app.

        Examples
        --------
        >>> class_cor = ClassCorrelation()
        >>> app = Dash(__name__)
        >>> class_cor._register_callbacks(app)
        """
        # Register the callback for updating the page content
        app.callback(
            Output(
                component_id='page-content-class-cor',
                component_property='children',
            ),
            Input(
                component_id='current-page-class-cor',
                component_property='data',
            ),
            Input(
                component_id='page-dropdown-class-cor',
                component_property='value',
            ),
            prevent_initial_call=True,
        )(self._display_page)

        # Register the callback for handling button clicks and updating
        app.callback(
            Output(
                component_id='current-page-class-cor',
                component_property='data',
            ),
            Output(
                component_id='page-dropdown-class-cor',
                component_property='value',
            ),
            Input(
                component_id='prev-page-class-cor',
                component_property='n_clicks',
            ),
            Input(
                component_id='next-page-class-cor',
                component_property='n_clicks',
            ),
            Input(
                component_id='page-dropdown-class-cor',
                component_property='value',
            ),
            prevent_initial_call=True,
        )(self._update_page)

        # Register the callback to update the dropdown style
        app.callback(
            Output(
                component_id='page-dropdown-class-cor',
                component_property='style',
            ),
            Input(
                component_id='page-dropdown-class-cor',
                component_property='value',
            ),
            prevent_initial_call=True,
        )(self._update_dropdown_style)

        # Register the callback to update the button styles
        app.callback(
            [
                Output(
                    component_id='prev-page-class-cor',
                    component_property='style',
                ),
                Output(
                    component_id='next-page-class-cor',
                    component_property='style',
                ),
            ],
            Input(
                component_id='page-dropdown-class-cor',
                component_property='value',
            ),
            prevent_initial_call=True,
        )(self._update_button_styles)

    def _display_page(
        self, current_page_key: str, selected_key: str
    ) -> html.Div:
        """
        Display the page based on the current page key and selected key.

        Name
        ----
        _display_page

        Description
        -----------
        This method checks if the selected key is in the basic statistics
        dictionary. If it is, it returns the corresponding layout. Otherwise,
        it returns a 404 page not found message.

        Dependencies
        ------------
            External
            --------
            dash
                Package for creating reactive web applications.

        Parameters
        ----------
        current_page_key : str
            The key of the current page.
        selected_key : str
            The selected key.

        Returns
        -------
        html.Div
            The HTML div containing the page content.

        Examples
        --------
        >>> class_cor = ClassCorrelation()
        >>> current_page_key = 'class1'
        >>> selected_key = 'class1'
        >>> page = class_cor._display_page(current_page_key,
        ...                                selected_key)
        >>> page
        html.Div(children=[...])
        """
        page_key = selected_key if selected_key else current_page_key

        if page_key in self.basic_correlation_dict:
            layout = self.basic_correlation_dict[page_key]('none')
        else:
            layout = html.Div(children="404 - Page Not Found")

        return layout

    def _update_page(
        self, prev_clicks: int, next_clicks: int, selected_key: str
    ) -> tuple[str, str]:
        """
        Update the page based on the previous and next clicks.

        Name
        ----
        _update_page

        Description
        -----------
        This method updates the page based on the previous and next clicks.
        It returns a tuple containing the updated page content and style.

        Dependencies
        ------------
            External
            --------
            dash
                Package for creating reactive web applications.

        Parameters
        ----------
        prev_clicks : int
            The number of previous clicks.
        next_clicks : int
            The number of next clicks.
        selected_key : str
            The currently selected key.

        Returns
        -------
        tuple[str, str]
            A tuple containing the updated page content and style.

        Examples
        --------
        >>> class_cor = ClassCorrelation()
        >>> prev_clicks = 1
        >>> next_clicks = 2
        >>> selected_key = 'class1'
        >>> page_content, page_style = class_cor._update_page(prev_clicks,
        ...                                                   next_clicks,
        ...                                                   selected_key
        ...                                                   )
        >>> page_content
        'This is the updated page content'
        >>> page_style
        'This is the updated page style'
        """
        ctx = dash.callback_context

        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            current_key = selected_key

            if button_id == 'next-page-class-cor':
                current_index = self.keys.index(current_key)
                if current_index < len(self.keys) - 1:
                    current_key = self.keys[current_index + 1]
            elif button_id == 'prev-page-class-cor':
                current_index = self.keys.index(current_key)
                if current_index > 0:
                    current_key = self.keys[current_index - 1]

            return current_key, current_key

        return selected_key, selected_key

    def _update_dropdown_style(self, selected_key: str) -> dict[str, str]:
        """
        Update the style of the dropdown menu based on the selected key.

        Name
        ----
        _update_dropdown_style


        Description
        -----------
        This method updates the style of the dropdown menu based on the
        selected key. It returns a dictionary containing the style for the
        dropdown menu.

        Parameters
        ----------
        selected_key : str
            The currently selected key.

        Returns
        -------
        dict[str, str]
            A dictionary containing the style for the dropdown menu.

        Examples
        --------
        >>> class_cor = ClassCorrelation()
        >>> selected_key = 'class1'
        >>> dropdown_style = class_cor._update_dropdown_style(selected_key)
        >>> dropdown_style
        {'width': '100%', 'color': '#000000', 'backgroundColor': '#ADD8E6'}
        """
        return {
            'width': '100%',
            'color': '#000000',
            'backgroundColor': self._lighten_color(
                color=selected_key, amount=0.3
            ),
        }

    def _update_button_styles(
        self, selected_key: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Update the styles of the buttons based on the selected key.

        Name
        ----
        _update_button_styles

        Description
        -----------
        This method updates the styles of the buttons based on the selected
        key. It returns two dictionaries containing the styles for the buttons.

        Parameters
        ----------
        selected_key : str
            The currently selected key.

        Returns
        -------
        tuple[dict[str, str], dict[str, str] ]
            A tuple of two dictionaries containing the styles for the buttons.

        Examples
        --------
        >>> class_statistics = ClassStatistics()
        >>> selected_key = 'class1'
        >>> button_styles = class_statistics._update_button_styles(
        ...                                         selected_key
        ...                                         )
        >>> button_styles
        ({'color': '#000000',
          'backgroundColor': '#ADD8E6'
          },
         {'color': '#000000',
          'backgroundColor': '#ADD8E6'
          })
        """
        color = self._lighten_color(color=selected_key, amount=0.3)
        button_style = {'color': '#000000', 'backgroundColor': color}
        return button_style, button_style

    def _notebook_display(self) -> None:
        """
        Display a dropdown widget that allows the user to select a class.

        Name
        ----
        _notebook_display

        Description
        -----------
        Creates a styled dropdown using ipywidgets that allows for selection
        of different classes, and updates the display based on the selected
        class.

        Dependencies
        ------------
            External
            --------
            ipywidgets
                Provides the interface elements (dropdown, output widget).
            ipykernel
                Used to display the dropdown and the output in the notebook.

        Examples
        --------
        >>> obj = YourClassName()
        >>> obj._notebook_display()
        """
        # Create a dropdown widget with improved styling and spacing
        dropdown = widgets.SelectionSlider(
            options=self.keys,
            description='Select Class: ',
            value=self.keys[0],
            style={
                'description_width': 'initial',
                'handle_color': self.keys[0],
            },  # Adjusts the description
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            layout=widgets.Layout(width='20%', margin_bottom='10px'),
        )

        output = widgets.Output(layout={'border': '2px solid #ddd'})

        # Set up the observer for the dropdown
        dropdown.observe(
            handler=lambda change: self._update_display(
                change=change, dropdown=dropdown, output=output
            ),
            names='value',
        )

        # Create a container for better layout
        container = widgets.VBox(
            children=[dropdown, output],
            layout=widgets.Layout(
                padding='20px', align_items='center', background_color='black'
            ),
        )

        # Display the dropdown and output widgets
        notebook_display(container, display_id=True)

        # Trigger initial update
        self._update_display(
            change={'new': self.keys[0]}, dropdown=dropdown, output=output
        )

    def _update_display(
        self,
        change: dict,
        dropdown: widgets.SelectionSlider,
        output: widgets.Output,
    ) -> None:
        """
        Update the output widget based on the selected class from the dropdown.

        Name
        ----
        _update_display

        Description
        -----------
        Clears the previous output and displays the Figure corresponding to
        the selected class. Updates the dropdown style to reflect the selected
        class.

        Dependencies
        ------------
            External
            --------
            ipywidgets
                Provides the interface elements (dropdown, output widget).

        Parameters
        ----------
        change : dict
            Contains information about the change event, particularly the new
            value selected in the dropdown.
        dropdown : widgets.SelectionSlider
            The dropdown widget for class selection.
        output : widgets.Output
            The output widget where the corresponding Figure will be
            displayed.

        Examples
        --------
        _update_display({'new': 'ClassA'}, dropdown, output)
        """
        # Clear the previous output
        with output:

            # Get the selected key
            selected_key = change['new']

            # Update dropdown style
            dropdown.style = {
                'description_width': 'initial',
                'handle_color': selected_key,
            }

            fig = self.basic_correlation_dict[selected_key]('notebook')
            notebook_display(fig)
            clear_output(wait=True)

    def __call__(self, display: str = 'container') -> Optional[html.Div]:
        """
        Display the class correlation in a Dash app or Jupyter notebook.

        Name
        ----
        __call__

        Description
        -----------
        This method displays the class correlation in a Dash app or Jupyter
        notebook. It provides an interactive interface for users to explore the
        statistics.

        Dependencies
        ------------
            External
            --------
            dash
                Package for making reactive web applications.
            dash_bootstrap_components
                Package for creating Bootstrap components.

        Parameters
        ----------
        display : str, optional
            The display mode. Can be 'notebook' or 'container'. Default is
            'container'.

        Examples
        --------
        >>> class_correlation = ClassCorrelation()
        >>> class_correlation()
        >>> class_correlation(display='notebook')
        """
        if display == 'notebook':
            self._notebook_display()
        elif display == 'container' or display == 'none':
            # Initialize the Dash app
            app = Dash(name=__name__, external_stylesheets=[dbc.themes.DARKLY])

            # Set the layout of the app
            app.layout = self._create_layout()

            # Register the callbacks
            self._register_callbacks(app)

            if display == 'container':
                # Run the Dash server on the specified port and host
                app.run(
                    debug=True,
                    host='0.0.0.0',
                    proxy="http://0.0.0.0:8050::https://127.0.0.1:8050",
                    port=8050,
                )
            elif display == 'none':
                return app.layout
        return None


if __name__ == "__main__":
    gen = GenerateData()
    split = SplitData(gen=gen)
    class_stats = ClassCorrelation(X=split.X.train, y=split.y.train)
    class_stats('container')
