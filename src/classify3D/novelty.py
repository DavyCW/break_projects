"""
Module calculates and displays novelty of a dataset.

Name
----
novelty.py

Description
-----------
This module provides a class for calculating and displaying novelty
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
    scikit-learn
        Package for machine learning algorithms.

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
    Novelty
        A class representing novelty of a dataset.
"""

import numpy as np
import pandas as pd
import dash
from dash import html, dcc
from src.classify3D.generate_data import GenerateData  # type: ignore
from src.classify3D.split_data import SplitData  # type: ignore
from typing import Optional, Union
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import ipywidgets as widgets
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


class Novelty:
    """
    A class representing novelty of a dataset.

    Name
    ----
    Novelty

    Description
    -----------
    This class provides methods for calculating novelty of a dataset.

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
        scikit-learn
            Package for machine learning algorithms.

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
            Initialize the Novelty object with a dataset.
        __repr__()
            Provide a string representation of the Novelty object.
        _get_novelty()
            Create a plotly figure to visualize the novelty.
        __call__()
            Display the novelty of the dataset.
        _get_z_score_novelty()
            Calculate the z-scores for each data point and identify novelty.
        _get_iqr_novelty()
            Calculate the interquartile range (IQR) and identify novelty.
        _get_isolation_forest_novelty()
            Identify novelty using the Isolation Forest algorithm.
        _get_one_class_svm_novelty()
            Identify novelty using the One-Class SVM algorithm.
        _get_local_outlier_factor_novelty()
            Identify novelty using the Local Outlier Factor (LOF) algorithm.

        Variables
        ---------
        X_train : np.ndarray
            The training data.
        X_test : np.ndarray
            The validation data.
        z_score : go.Figure
            A Go Figure containing the z_score of the dataset.
        iqr : go.Figure
            A Go Figure containing the iqr of the dataset.
        isolation_forest : go.Figure
            A Go Figure containing the isolation_forest of the dataset.
        one_class_svm : go.Figure
            A Go Figure containing the one_class_svm of the dataset.
        local_outlier_factor : go.Figure
            A Go Figure containing the local_outlier_factor of the dataset.
    """

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray) -> None:
        """
        Initialize the Novelty object with a dataset.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the Novelty object with a dataset.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        X_train : np.ndarray
            The feature data.
        X_test : np.ndarray
            The feature data.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        """
        self.X_train = X_train
        self.X_test = X_test
        self.z_score = self._get_novelty(
            method='z_score', novelty=self._get_z_score_novelty()
        )
        self.iqr = self._get_novelty(
            method='iqr', novelty=self._get_iqr_novelty()
        )
        self.isolation_forest = self._get_novelty(
            method='isolation_forest',
            novelty=self._get_isolation_forest_novelty(),
        )
        self.one_class_svm = self._get_novelty(
            method='one_class_svm', novelty=self._get_one_class_svm_novelty()
        )
        self.local_outlier_factor = self._get_novelty(
            method='local_outlier_factor',
            novelty=self._get_local_outlier_factor_novelty(),
        )

    def _get_z_score_novelty(self, threshold: float = 3) -> np.ndarray:
        """
        Calculate the z-scores for each data point and identify novelty.

        Name
        ----
        _get_z_score_outliers

        Description
        -----------
        This method calculates the z-scores for each data point in the dataset
        and identifies novelty based on a given threshold.

        Dependencies
        ------------
            External
            --------
            stats
                Package for statistical distributions.
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        threshold : float, optional
            The threshold value for identifying novelty (default is 3).

        Returns
        -------
        np.ndarray
            A boolean array indicating whether each data point is a novelty.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty._get_z_score_novelty(threshold=2)
        """
        X_train_mean = np.mean(a=self.X_train, axis=0)
        X_train_std = np.std(a=self.X_train, axis=0)
        z_scores = np.abs((self.X_test - X_train_mean) / X_train_std)
        return (z_scores > threshold).any(axis=1)

    def _get_iqr_novelty(self, threshold: float = 1.5) -> np.ndarray:
        """
        Calculate the interquartile range (IQR) and identify novelty.

        Name
        ----
        _get_iqr_outliers

        Description
        -----------
        This method calculates the IQR for each feature in the dataset and
        identifies novelty based on a given threshold.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        threshold : float, optional
            The threshold value for identifying novelty (default is 1.5).

        Returns
        -------
        np.ndarray
            A boolean array indicating whether each data point is a novelty.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty._get_iqr_novelty(threshold=2)
        """
        Q1 = np.percentile(a=self.X_train, q=25, axis=0)
        Q3 = np.percentile(a=self.X_train, q=75, axis=0)
        iqr = Q3 - Q1
        lower_bound = Q1 - threshold * iqr
        upper_bound = Q3 + threshold * iqr
        return np.any(
            a=(self.X_test < lower_bound) | (self.X_test > upper_bound), axis=1
        )

    def _get_isolation_forest_novelty(
        self, contamination: Union[str, float] = 'auto'
    ) -> np.ndarray:
        """
        Identify novelty using the Isolation Forest algorithm.

        Name
        ----
        _get_isolation_forest_novelty

        Description
        -----------
        This method uses the Isolation Forest algorithm to identify novelty in
        the dataset.

        Dependencies
        ------------
            External
            --------
            scikit-learn
                Package for machine learning.
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        contamination : str | float, optional
            The proportion of novelty in the data (default is 'auto').

        Returns
        -------
        np.ndarray
            A boolean array indicating whether each data point is a novelty.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty._get_isolation_forest_novelty(contamination=0.1)
        """
        model = IsolationForest(contamination=contamination)
        model.fit(X=self.X_train)
        return model.decision_function(X=self.X_test) < 0

    def _get_one_class_svm_novelty(self, nu: float = 0.05) -> np.ndarray:
        """
        Identify novelty using the One-Class SVM algorithm.

        Name
        ----
        _get_one_class_svm_outliers

        Description
        -----------
        This method uses the One-Class SVM algorithm to identify novelty in
        the dataset.

        Dependencies
        ------------
            External
            --------
            scikit-learn
                Package for machine learning.
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        nu : float, optional
            The parameter that controls the number of support vectors (default
            is 0.05).

        Returns
        -------
        np.ndarray
            A boolean array indicating whether each data point is a novelty.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty._get_one_class_svm_novelty(nu=0.1)
        """
        model = OneClassSVM(gamma='auto', nu=nu)  # nu can be tuned
        model.fit(X=self.X_train)
        return model.decision_function(X=self.X_test) < 0

    def _get_local_outlier_factor_novelty(
        self, n_neighbors: int = 20
    ) -> np.ndarray:
        """
        Identify novelty using the Local Outlier Factor (LOF) algorithm.

        Name
        ----
        _get_local_outlier_factor_novelty

        Description
        -----------
        This method uses the LOF algorithm to identify novelty in the dataset.
        The LOF algorithm calculates the local density of each data point and
        identifies points with a density significantly lower than their
        neighbors as novelty.

        Dependencies
        ------------
            External
            --------
            scikit-learn
                Package for machine learning.
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        n_neighbors : int, optional
            The number of neighbors to consider when calculating the local
            density (default is 20).

        Returns
        -------
        np.ndarray
            A boolean array indicating whether each data point is a novelty.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty._get_local_outlier_factor_novelty(n_neighbors=10)
        """
        model = LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors)
        model.fit(X=self.X_train)
        return model.decision_function(X=self.X_test) < 0

    def _get_novelty(self, method: str, novelty: np.ndarray) -> go.Figure:
        """
        Create a plotly figure to visualize the novelty.

        Name
        ----
        _get_outliers

        Description
        -----------
        This method creates a plotly figure to visualize the novelty
        identified by the specified method.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.
            plotly
                Package for creating interactive plots.
            pandas
                Package for data manipulation and analysis.

        Parameters
        ----------
        method : str
            The method used to identify outliers.
        outliers : np.ndarray
            A boolean array indicating whether each data point is an outlier.

        Returns
        -------
        go.Figure
            A plotly figure visualizing the novelty.

        Examples
        --------
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty._get_novelty(
        ...     method='LOF', novelty=np.array(
        ...         [True, False, True, False, True]
        ...         )
        ...     )
        """
        # Create a DataFrame for Plotly
        df_plot = pd.DataFrame(
            data=self.X_test,
            columns=[f'Feature {i+1}' for i in range(self.X_test.shape[1])],
        )
        df_plot['Novelty'] = novelty

        # Create figure
        fig = go.Figure(
            data=go.Scatter3d(
                x=df_plot.loc[df_plot['Novelty'], 'Feature 1'],
                y=df_plot.loc[df_plot['Novelty'], 'Feature 2'],
                z=df_plot.loc[df_plot['Novelty'], 'Feature 3'],
                mode='markers',
                marker=dict(size=1, opacity=0.4, color='red', symbol='x'),
                name='Novelty',
            )
        )

        # Update layout
        fig.update_layout(
            title={
                'text': f'3D Scatter Plot of Features with {method} Novelty',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 16,
                },
            },
            plot_bgcolor='#222',  # Background color for the plot area
            paper_bgcolor='#222',  # Background color for the paper
            font=dict(color='white'),
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                bgcolor='#222',
                xaxis=dict(
                    showgrid=False, backgroundcolor='#222', color='#222'
                ),  # Disable x-axis grid
                yaxis=dict(
                    showgrid=False, backgroundcolor='#222', color='#222'
                ),  # Disable y-axis grid
                zaxis=dict(
                    showgrid=False, backgroundcolor='#222', color='#222'
                ),  # Disable z-axis grid
            ),
            showlegend=False,
        )
        return fig

    def __repr__(self) -> str:
        """
        Provide a string representation of the Novelty object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the Novelty object.

        Returns
        -------
        str
            A string representation of the Novelty object.

        Examples
        --------
        >>> print(novelty)
        Novelty()
        """
        return f"Novelty(X_test.shape={self.X_test.shape})"

    def __call__(self, display: str = 'notebook') -> Optional[html.Div]:
        """
        Display the novelty of the dataset.

        Name
        ----
        __call__

        Description
        -----------
        Display the outliers of the dataset in a Jupyter notebook or as a Dash
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
        >>> novelty = Novelty(X_train=np.array([1, 2, 3, 4, 5]),
        ...                   X_test=np.array([1, 2, 3, 4, 5])
        ...                   )
        >>> novelty()
        >>> novelty(display='container')
        """
        if display == 'notebook':
            # Convert fig to a FigureWidget
            figure_widgets = [
                go.FigureWidget(data=fig)
                for fig in [
                    self.z_score,
                    self.iqr,
                    self.isolation_forest,
                    self.one_class_svm,
                    self.local_outlier_factor,
                ]
            ]
            container = widgets.VBox(
                children=figure_widgets,
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
                            children=[
                                dcc.Graph(id='z_score', figure=self.z_score),
                                dcc.Graph(id='iqr', figure=self.iqr),
                                dcc.Graph(
                                    id='isolation_forest',
                                    figure=self.isolation_forest,
                                ),
                                dcc.Graph(
                                    id='one_class_svm',
                                    figure=self.one_class_svm,
                                ),
                                dcc.Graph(
                                    id='local_outlier_factor',
                                    figure=self.local_outlier_factor,
                                ),
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
    novelty = Novelty(X_train=split.X.train, X_test=split.X.test)
    novelty('container')
