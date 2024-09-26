"""
Module calculates and displays basic correlation of a dataset.

Name
----
basic_correlation.py

Description
-----------
This module provides a class for calculating and displaying basic correlation
of a dataset.

Dependencies
------------
    External
    --------
    numpy
        A library for efficient numerical computation.
    pandas
        Package for data manipulation and analysis.
    dash
        Package for making reactive web applications.
    ipykernel
        Package for displaying plots in Jupyter notebooks.
    matplotlib
        Package for good color distributions.
    dash_bootstrap_components
        Package for creating Bootstrap components.
    plotly
        Package to plot 3D data beautifully.

    Internal
    --------
    GenerateData
        Generates 3D data based on passed file.
    SplitData
        A class responsible for splitting the data.

Attributes
----------
    Classes
    -------
    BasicCorrelation
        A class representing basic correlation of a dataset.
"""

import numpy as np
import pandas as pd
import dash
from dash import html, dcc
from src.classify3D.generate_data import GenerateData  # type: ignore
from src.classify3D.split_data import SplitData  # type: ignore
from typing import Optional
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


class BasicCorrelation:
    """
    A class representing basic correlation of a dataset.

    Name
    ----
    BasicCorrelation

    Description
    -----------
    This class provides methods for calculating basic correlation of a dataset.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pandas
            Package for data manipulation and analysis.
        dash
            Package for making reactive web applications.
        matplotlib
            Package for good color distributions.
        dash_bootstrap_components
            Package for creating Bootstrap components.
        plotly
            Package to plot 3D data beautifully.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.
        SplitData
            A class responsible for splitting the data.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the BasicStatistics object with a dataset.
        __repr__()
            Provide a string representation of the BasicStatistics object.
        _get_basic_correlation()
            Calculate basic correlation of a dataset.
        __call__()
            Display the basic statistics of the dataset.

        Variables
        ---------
        basic_correlation : go.Figure
            A Go Figure containing the basic correlation of the dataset.
    """

    def __init__(self, X: np.ndarray, name: str) -> None:
        """
        Initialize the BasicCorrelation object with a dataset.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the BasicCorrelation object with a dataset.

        Dependencies
        ------------
            External
            --------
            numpy
                A library for efficient numerical computation.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        name : str
            The name of the dataset.

        Examples
        --------
        >>> basic_correlation = BasicCorrelation(X=np.array([1, 2, 3, 4, 5]),
        ...                                    name='my_dataset'
        ...                                    )
        """
        self.base_color = name

        self.basic_correlation = self._get_basic_correlation(X=X)

    def __repr__(self) -> str:
        """
        Provide a string representation of the BasicCorrelation object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the BasicCorrelation object.

        Returns
        -------
        str
            A string representation of the BasicCorrelation object.

        Examples
        --------
        >>> print(basic_correlation)
        BasicCorrelation(base_color='my_dataset')
        """
        return f"BasicCorrelation(base_color={self.base_color})"

    def _get_basic_correlation(self, X: np.ndarray) -> go.Figure:
        """
        Calculate basic correlation of a dataset.

        Name
        ----
        _get_basic_correlation

        Description
        -----------
        Calculate basic correlation of a dataset.

        Dependencies
        ------------
            External
            --------
            numpy
                A library for efficient numerical computation.
            pandas
                Package for data manipulation and analysis.
            plotly
                Package to plot 3D data beautifully.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        Returns
        -------
        go.Figure
            A Go Figure containing the basic correlation of the dataset.

        Examples
        --------
        >>> basic_correlation = BasicCorrelation._get_basic_correlation(
        ...                             X=np.array([1, 2, 3, 4, 5]),
        ...                             )
        """
        # Create a Pandas DataFrame
        df = pd.DataFrame(data=X, columns=['x', 'y', 'z', 'r', 'theta', 'phi'])

        # Calculate the correlation matrix
        corr_matrix = df.corr().iloc[::-1, :]

        # Define your custom hex color and white as the color scale
        custom_colorscale = [[0, '#FFFFFF'], [1, self.base_color]]

        # Create the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=custom_colorscale,
                reversescale=False,
                showscale=True,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
            )
        )

        # Update the layout
        fig.update_layout(
            title={
                'text': f"{self.base_color} Correlation",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 28,
                    'color': self.base_color,
                },
            },
            xaxis_title='Features',
            yaxis_title='Features',
            font_size=14,
            width=750,
            height=750,
            font=dict(color=self.base_color),
            paper_bgcolor='#222',
            autosize=False,
            xaxis=dict(
                scaleanchor='y', constrain='domain', color=self.base_color
            ),
            yaxis=dict(
                scaleanchor='x', constrain='domain', color=self.base_color
            ),
        )
        return fig

    def __call__(self, display: str = 'notebook') -> Optional[html.Div]:
        """
        Display the basic correlation of the dataset.

        Name
        ----
        __call__

        Description
        -----------
        Display the basic correlation of the dataset in a Jupyter notebook or
        as a Dash app.

        Dependencies
        ------------
            External
            --------
            dash
                Package for making reactive web applications.
            dash_bootrstrap_components
                Package for creating Bootstrap components.

        Parameters
        ----------
        display : str, optional
            The display mode. Can be 'notebook' or 'container'. By default
            'notebook'.

        Raises
        ------
        ValueError
            If the display mode is not 'notebook' or 'container'.

        Examples
        --------
        >>> basic_correlation = BasicCorrelation(X=np.array([1, 2, 3, 4, 5]),
        ...                                      name='my_dataset'
        ...                                      )
        >>> basic_correlation()
        >>> basic_correlation(display='container')
        """
        if display == 'notebook':
            # Display the DataFrame within a Jupyter notebook
            self.basic_correlation.show()

        elif display == 'container' or display == 'none':
            app = dash.Dash(
                name=__name__, external_stylesheets=[dbc.themes.DARKLY]
            )

            # Define the layout of the Dash app
            app.layout = dbc.Container(
                children=[
                    dbc.Row(
                        children=dbc.Col(
                            children=dcc.Graph(
                                id='heatmap', figure=self.basic_correlation
                            )
                        )
                    ),
                ],
                fluid=True,
                style={
                    'display': 'flex',
                    'flex-direction': 'column',
                    'align-items': 'center',
                    'justify-content': 'flex-start',
                    'height': '100vh',
                    'padding-top': '20px',
                },
            )

            if display == 'container':
                # Run the Dash server on the specified port and host
                app.run_server(debug=True, host='0.0.0.0', port=8050)
            elif display == 'none':
                return app.layout

        else:
            raise ValueError(
                f"Invalid display option '{display}'. "
                f"Use 'notebook', 'container', or 'none'."
            )
        return None


if __name__ == "__main__":
    gen = GenerateData()
    split = SplitData(gen=gen)
    basic_cor = BasicCorrelation(split.X.train, name='#00b5eb')
    basic_cor('container')
