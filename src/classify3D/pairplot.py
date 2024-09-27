"""
Module calculates and displays pairplot of a dataset.

Name
----
pairplot.py

Description
-----------
This module provides a class for calculating and displaying pairplot
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
    matplotlib
        Package for good color distributions.
    dash_bootstrap_components
        Package for creating Bootstrap components.
    plotly
        Package to plot 3D data beautifully.
    ipywidgets
        Package for making interactive web applications.

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
    PairPlot
        A class representing pairplot of a dataset.
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
import ipywidgets as widgets
from plotly.subplots import make_subplots


class PairPlot:
    """
    A class representing pairplot of a dataset.

    Name
    ----
    PairPlot

    Description
    -----------
    This class provides methods for calculating pairplot of a dataset.

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
        ipywidgets
            Package for making interactive web applications.

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
            Initialize the PairPlot object with a dataset.
        __repr__()
            Provide a string representation of the PairPlot object.
        _get_pairplot()
            Calculate pairplot of a dataset.
        __call__()
            Display the pairplot of the dataset.

        Variables
        ---------
        sample_size : int
            The number of samples to display in the pairplot.
        pairplot : go.Figure
            A Go Figure containing the pairplot of the dataset.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, sample_size: int = 2560
    ) -> None:
        """
        Initialize the PairPlot object with a dataset.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the PairPlot object with a dataset.

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
        sample_size : int
            The number of samples to display in the pairplot.

        Examples
        --------
        >>> pairplot = PairPlot(X=np.array([1, 2, 3, 4, 5]),
        ...                     y=np.array([0, 0, 1, 1, 1])
        ...                     )
        """
        self.sample_size = sample_size
        self.pairplot = self._get_pairplot(X=X, y=y)

    def __repr__(self) -> str:
        """
        Provide a string representation of the PairPlot object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the PairPlot object.

        Returns
        -------
        str
            A string representation of the PairPlot object.

        Examples
        --------
        >>> print(pairplot)
        PairPlot()
        """
        return "PairPlot()"

    def _get_pairplot(self, X: np.ndarray, y: np.ndarray) -> go.Figure:
        """
        Calculate pairplot of a dataset.

        Name
        ----
        _get_pairplot

        Description
        -----------
        Calculate pairplot of a dataset.

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
            matplotlib
                Package for good color distributions.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray
            The label data.

        Returns
        -------
        go.Figure
            A Go Figure containing the pairplot of the dataset.

        Examples
        --------
        >>> pairplot = PairPlot._get_pairplot(
        ...                             X=np.array([1, 2, 3, 4, 5]),
        ...                             y=np.array([0, 0, 1, 1, 1])
        ...                             )
        """
        indices = np.random.choice(X.shape[0], self.sample_size, replace=False)

        # Create a DataFrame from the sampled data
        sampled_X = X[indices]
        sampled_y = y[indices]

        # Feature names
        feature_names = ['x', 'y', 'z', 'r', 'theta', 'phi']

        # Create a DataFrame for sampled data
        df = pd.DataFrame(data=sampled_X, columns=feature_names)
        df['color'] = sampled_y  # Add the hex color codes to the DataFrame

        # Create a grid of subplots for pairplot
        fig = make_subplots(
            rows=len(feature_names),
            cols=len(feature_names),
            subplot_titles=[
                f'{feature_names[i]} vs {feature_names[j]}' if i >= j else ''
                for i in range(len(feature_names))
                for j in range(len(feature_names))
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,  # Adjust spacing to make plots more square
        )

        # Add scatter plots (only in the lower triangle) and colored density
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if i == j:
                    # Create a density plot for each category using Histogram
                    for color in np.unique(df['color']):
                        # Filter the DataFrame for the current color
                        filtered_df = df[df['color'] == color]
                        fig.add_trace(
                            trace=go.Histogram(
                                x=filtered_df[feature_names[i]],
                                histnorm='probability density',
                                opacity=0.5,
                                name=f'Density {color}',
                                marker=dict(color=color),
                                hoverinfo='none',
                                # Reduce the number of bins
                                xbins=dict(
                                    start=filtered_df[feature_names[i]].min(),
                                    end=filtered_df[feature_names[i]].max(),
                                    size=(
                                        filtered_df[feature_names[i]].max()
                                        - filtered_df[feature_names[i]].min()
                                    )
                                    / 30,
                                ),  # Example for fewer bins
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                elif i > j:
                    # Add scatter plot in the lower triangle
                    fig.add_trace(
                        go.Scatter(
                            x=df[feature_names[j]],
                            y=df[feature_names[i]],
                            mode='markers',
                            marker=dict(
                                color=df['color'],
                                size=3,
                                line=dict(width=0.5, color='DarkSlateGrey'),
                            ),
                            showlegend=False,
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

        # Update layout for the overall figure
        fig.update_layout(
            title={
                'text': 'Sampled Pairplot with Density Plots Colored by Y',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 16,
                },
            },
            width=1200,
            height=1200,
            plot_bgcolor='#222',  # Background color for the plot area
            paper_bgcolor='#222',  # Background color for the paper
            font=dict(color='white'),
            showlegend=False,
        )

        # Update axes titles and remove formatting from upper triangle
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if i == j:
                    fig.update_xaxes(
                        title_text=feature_names[j], row=i + 1, col=j + 1
                    )
                    fig.update_yaxes(
                        title_text=feature_names[i], row=i + 1, col=j + 1
                    )
                elif i < j:
                    # Hide axes for the upper triangle
                    fig.update_xaxes(
                        showticklabels=False, row=i + 1, col=j + 1
                    )
                    fig.update_yaxes(
                        showticklabels=False, row=i + 1, col=j + 1
                    )
        return fig

    def __call__(self, display: str = 'notebook') -> Optional[html.Div]:
        """
        Display the pairplot of the dataset.

        Name
        ----
        __call__

        Description
        -----------
        Display the pairplot of the dataset in a Jupyter notebook or as a Dash
        app.

        Dependencies
        ------------
            External
            --------
            dash
                Package for making reactive web applications.
            dash_bootrstrap_components
                Package for creating Bootstrap components.
            ipywidgets
                Package for making interactive web applications.

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
        >>> pairplot = PairPlot(X=np.array([1, 2, 3, 4, 5]),
        ...                     y=np.array([0, 0, 1, 1, 1])
        ...                     )
        >>> pairplot()
        >>> pairplot(display='container')
        """
        if display == 'notebook':
            # Convert fig to a FigureWidget
            fig_widget = go.FigureWidget(data=self.pairplot)

            # Create a VBox container to hold the figure widget
            container = widgets.VBox(
                children=[fig_widget],
                layout=widgets.Layout(align_items='center'),
            )
            return container

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
                                id='heatmap', figure=self.pairplot
                            ),
                            width="auto",
                        ),
                        justify='center',  # Center the Row content
                    ),
                ],
                fluid=True,
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
    pairplot = PairPlot(split.X.train, y=split.y.train)
    pairplot('container')
