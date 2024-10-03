"""
Module calculates and displays distributions of a dataset.

Name
----
basic_correlation.py

Description
-----------
This module provides a class for calculating and displaying distributions
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
    scipy
        Package for statistical distributions.

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
    Distributions
        A class representing distributions of a dataset.
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
import plotly.express as px
from scipy import stats


class Distributions:
    """
    A class representing distributions of a dataset.

    Name
    ----
    Distributions

    Description
    -----------
    This class provides methods for calculating distributions of a dataset.

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
        scipy
            Package for statistical distributions.

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
            Initialize the Distributions object with a dataset.
        __repr__()
            Provide a string representation of the Distributions object.
        _get_pairplot()
            Calculate distributions of a dataset.
        __call__()
            Display the distributions of the dataset.

        Variables
        ---------
        bar_plot : go.Figure
            A Go Figure containing the distributions of the dataset.
        histogram : go.Figure
            A Go Figure containing the distributions of the dataset.
        kde : go.Figure
            A Go Figure containing the distributions of the dataset.
    """

    def __init__(self, y: np.ndarray) -> None:
        """
        Initialize the Distributions object with a dataset.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the Distributions object with a dataset.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        y : np.ndarray
            The label data.

        Examples
        --------
        >>> distributions = Distributions(y=np.array([0, 0, 1, 1, 1]))
        """
        self.bar_plot, self.histogram, self.kde = self._get_distributions(y=y)

    def __repr__(self) -> str:
        """
        Provide a string representation of the Distributions object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the Distributions object.

        Returns
        -------
        str
            A string representation of the Distributions object.

        Examples
        --------
        >>> print(distributions)
        Distributions()
        """
        return "Distributions()"

    def _update_layout(
        self,
        fig: go.Figure,
        title_text: str,
        xaxis_title: str,
        yaxis_title: str,
    ) -> go.Figure:
        """
        Update the layout of a plotly figure.

        Name
        ----
        _update_layout

        Description
        -----------
        Update the layout of a plotly figure.

        Dependencies
        ------------
            External
            --------
            plotly
                Package to plot 3D data beautifully.

        Parameters
        ----------
        fig : go.Figure
            _description_
        title_text : str
            Title of the plot.
        xaxis_title : str
            Name of the x-axis.
        yaxis_title : str
            Name of the y-axis.

        Returns
        -------
        go.Figure
            Updated plotly figure.

        Examples
        --------
        _example_
        """
        fig.update_layout(
            showlegend=False,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font=dict(color='white'),
            bargap=0,
            height=400,
            width=800,
            title={
                'text': title_text,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16},
            },
        )
        return fig

    def _create_bar_plot(
        self, data: pd.Series, x_col: str, y_col: str, title: str
    ) -> go.Figure:
        """
        Create a bar plot from labels.

        Name
        ----
        _create_bar_plot

        Description
        -----------
        Create a bar plot from labels.

        Dependencies
        ------------
            External
            --------
            plotly
                Package to plot 3D data beautifully.
            pandas
                Package for data manipulation and analysis.

        Parameters
        ----------
        data : pd.Series
            Label data.
        x_col : str
            Name of the column to be used as the x-axis.
        y_col : str
            Name of the column to be used as the y-axis.
        title : str
            Title of the plot.

        Returns
        -------
        go.Figure
            Bar plot of labels.

        Examples
        --------
        _example_
        """
        fig = px.bar(data_frame=data, x=x_col, y=y_col, color=x_col)
        fig = self._update_layout(
            fig=fig,
            title_text=title,
            xaxis_title='Hex Codes',
            yaxis_title='Count',
        )
        fig.update_xaxes(showticklabels=False)  # Hide x-axis labels if needed
        return fig

    def _create_histogram(
        self, data: pd.Series, x_col: str, title: str, nbins: int = 64
    ) -> go.Figure:
        """
        Create histogram figure.

        Name
        ----
        _create_histogram

        Description
        -----------
        Create histogram figure.

        Dependencies
        ------------
            External
            --------
            plotly
                Package to plot 3D data beautifully.
            pandas
                Package for data manipulation and analysis.

        Parameters
        ----------
        data : pd.Series
            Label data.
        x_col : str
            Name of the column to be used as the x-axis.
        title : str
            Title of the plot.
        nbins : int, optional
            Number of bins, by default 64

        Returns
        -------
        go.Figure
            Histogram of labels.

        Examples
        --------
        _example_
        """
        fig = px.histogram(data_frame=data, color=x_col, nbins=nbins)
        fig = self._update_layout(
            fig=fig,
            title_text=title,
            xaxis_title='Count',
            yaxis_title='Bin Size',
        )
        return fig

    def _create_kde_plot(self, data: pd.Series, title: str) -> go.Figure:
        """
        Create a kde plot.

        Name
        ----
        _create_kde_plot

        Description
        -----------
        Create a kde plot.

        Dependencies
        ------------
            External
            --------
            scipy
                Package for statistical distributions.
            numpy
                Package for fast array manipulation.
            plotly
                Package to plot 3D data beautifully.
            pandas
                Package for data manipulation and analysis.

        Parameters
        ----------
        data : pd.Series
            Label data.
        title : str
            Title of the plot.

        Returns
        -------
        go.Figure
            KDE plot of labels.

        Examples
        --------
        _example_
        """
        kde = stats.gaussian_kde(dataset=data['Count'])
        x_values = np.linspace(data['Count'].min(), data['Count'].max(), 500)
        kde_values = kde(x_values)

        fig = go.Figure()
        fig.add_trace(
            trace=go.Scatter(
                x=x_values,
                y=kde_values,
                mode='lines',
                line=dict(color='cyan', width=2),
                name='KDE',
            )
        )
        fig = self._update_layout(
            fig=fig,
            title_text=title,
            xaxis_title='Count',
            yaxis_title='Density',
        )
        return fig

    def _get_distributions(
        self, y: np.ndarray
    ) -> tuple[go.Figure, go.Figure, go.Figure]:
        """
        Calculate distributions of a dataset.

        Name
        ----
        _get_distributions

        Description
        -----------
        Calculate distributions of a dataset.

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
        y : np.ndarray
            The label data.

        Returns
        -------
        tuple[go.Figure, go.Figure, go.Figure ]
            3 Go Figures containing the distributions of the dataset.

        Examples
        --------
        >>> distributions = Distributions()._get_distributions(
        ...                             y=np.array([0, 0, 1, 1, 1])
        ...                             )
        """
        # Step 1: Count the occurrences of each unique hex code
        hex_counts = pd.Series(data=y).value_counts().reset_index()
        hex_counts.columns = ['Hex Code', 'Count']
        # Create the bar plot, histogram, and KDE plot widgets
        bar_plot = self._create_bar_plot(
            data=hex_counts,
            x_col='Hex Code',
            y_col='Count',
            title='Count of Unique Hex Codes',
        )
        histogram = self._create_histogram(
            data=hex_counts,
            x_col='Hex Code',
            title='Hex Codes Binned by Count',
        )

        kde = self._create_kde_plot(
            data=hex_counts, title='Approximation of Count Distribution (KDE)'
        )

        return bar_plot, histogram, kde

    def __call__(self, display: str = 'notebook') -> Optional[html.Div]:
        """
        Display the distributions of the dataset.

        Name
        ----
        __call__

        Description
        -----------
        Display the distributions of the dataset in a Jupyter notebook or as a
        Dash app.

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
            plotly
                Package to plot 3D data beautifully.

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
        >>> distributions = Distributions(X=np.array([1, 2, 3, 4, 5]),
        ...                     y=np.array([0, 0, 1, 1, 1])
        ...                     )
        >>> distributions()
        >>> distributions(display='container')
        """
        if display == 'notebook':
            # Create the bar plot, histogram, and KDE plot widgets
            fig_widget_b = go.FigureWidget(data=self.bar_plot)
            fig_widget_h = go.FigureWidget(data=self.histogram)
            fig_widget_d = go.FigureWidget(data=self.kde)

            # Create a VBox container to hold the figure widgets
            return widgets.VBox(
                children=[fig_widget_b, fig_widget_h, fig_widget_d],
                layout=widgets.Layout(align_items='center'),
            )

        elif display == 'container' or display == 'none':
            app = dash.Dash(
                name=__name__, external_stylesheets=[dbc.themes.DARKLY]
            )
            # Define the layout of the Dash app
            app.layout = dbc.Container(
                children=[
                    dbc.Row(
                        children=dbc.Col(
                            children=[
                                dcc.Graph(
                                    id='bar plot-dist', figure=self.bar_plot
                                ),
                                dcc.Graph(
                                    id='histogram-dist', figure=self.histogram
                                ),
                                dcc.Graph(id='kde-dist', figure=self.kde),
                            ],
                            width="auto",
                        ),
                        justify='center',  # Center the Row content
                    ),
                ],
                fluid=True,
            )

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

        else:
            raise ValueError(
                f"Invalid display option '{display}'. "
                f"Use 'notebook', 'container', or 'none'."
            )
        return None


if __name__ == "__main__":
    gen = GenerateData()
    split = SplitData(gen=gen)
    distribution = Distributions(y=split.y.train)
    distribution('container')
