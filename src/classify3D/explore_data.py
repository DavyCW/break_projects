"""
Module provides a class for performing exploratory data analysis.

Name
----
explore_data.py

Description
-----------
This module provides a class for exploring data.

Dependencies
------------
    External
    --------
    dash
        Package for making reactive web applications.
    dash_bootstrap_components
        Package for creating Bootstrap components.

    Internal
    --------
    generate_data
        A class for generating data.
    split_data
        A class responsible for splitting the data.
    basic_statistics
        A class for displaying basic statistics.
    class_statistics
        A class for displaying class statistics.
    basic_correlation
        A class for displaying basic correlation.
    class_correlation
        A class for displaying class correlation.
    pairplot
        A class for displaying pairplot.
    distributions
        A class for displaying distributions.
    outliers
        A class for displaying outliers.
    novelty
        A class for displaying novelty.

Attributes
----------
    Classes
    -------
    ExploreData
        Wrapper class for exploring data.
"""

from src.classify3D import (
    generate_data,
    split_data,
    basic_statistics,
    class_statistics,
    basic_correlation,
    class_correlation,
    pairplot,
    distributions,
    outliers,
    novelty,
)
import dash
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


class ExploreData:
    """
    Wrapper class for exploring data.

    Name
    ----
    ExploreData

    Description
    -----------
    Wrapper class for exploring data.

    Dependencies
    ------------
        External
        --------
        dash
            Package for making reactive web applications.
        dash_bootstrap_components
            Package for creating Bootstrap components.

        Internal
        --------
        generate_data
            A class for generating data.
        split_data
            A class responsible for splitting the data.
        basic_statistics
            A class for displaying basic statistics.
        class_statistics
            A class for displaying class statistics.
        basic_correlation
            A class for displaying basic correlation.
        class_correlation
            A class for displaying class correlation.
        pairplot
            A class for displaying pairplot.
        distributions
            A class for displaying distributions.
        outliers
            A class for displaying outliers.
        novelty
            A class for displaying novelty.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the ExploreData object with a SplitData object.
        __repr__()
            Provide a string representation of the ExploreData object.
        _register_callbacks()
            Register the callbacks for the Dash app.
        _setup_layout()
            Set up the layout for the Dash app.
        _render_content()
            Render the content based on the selected tab.
        __call__()
            Display in a Dash app or Jupyter notebook.

        Variables
        ---------
        gen : generate_data.GenerateData
            The GenerateData object.
        basic_stats_train : basic_statistics.BasicStatistics
            The BasicStatistics object for the train data.
        basic_stats_test : basic_statistics.BasicStatistics
            The BasicStatistics object for the test data.
        basic_stats_val : basic_statistics.BasicStatistics
            The BasicStatistics object for the val data.
        class_stats : class_statistics.ClassStatistics
            The ClassStatistics object for the dataset.
        basic_cor_train : basic_correlation.BasicCorrelation
            The BasicCorrelation object for the train data.
        basic_cor_test : basic_correlation.BasicCorrelation
            The BasicCorrelation object for the test data.
        basic_cor_val : basic_correlation.BasicCorrelation
            The BasicCorrelation object for the val data.
        class_cor : class_correlation.ClassCorrelation
            The ClassCorrelation object for the dataset.
        pplot : pairplot.PairPlot
            The PairPlot object for the dataset.
        dis : distributions.Distributions
            The Distributions object for the dataset.
        out : outliers.Outliers
            The Outliers object for the dataset.
        nov_test : novelty.Novelty
            The Novelty object for the test data.
        nov_val : novelty.Novelty
            The Novelty object for the val data.
        layouts : dict
            A dictionary containing the layouts for the Dash app.
    """

    def __init__(self, split: split_data.SplitData) -> None:
        """
        Initialize the ExploreData object with a SplitData object.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the ExploreData object with a SplitData object.

        Dependencies
        ------------
            Internal
            --------
            generate_data
                A class for generating data.
            split_data
                A class responsible for splitting the data.
            basic_statistics
                A class for displaying basic statistics.
            class_statistics
                A class for displaying class statistics.
            basic_correlation
                A class for displaying basic correlation.
            class_correlation
                A class for displaying class correlation.
            pairplot
                A class for displaying pairplot.
            distributions
                A class for displaying distributions.
            outliers
                A class for displaying outliers.
            novelty
                A class for displaying novelty.

        Parameters
        ----------
        split : split_data.SplitData
            The SplitData object containing the data to be explored.

        Examples
        --------
        >>> gen = generate_data.GenerateData()
        >>> split = split_data.SplitData(gen=gen)
        >>> explore = ExploreData(split=split)
        """
        self.gen = split.gen
        self.basic_stats_train = basic_statistics.BasicStatistics(
            X=split.X.train, name='#00b5eb'
        )
        self.basic_stats_test = basic_statistics.BasicStatistics(
            X=split.X.test, name='#00b5eb'
        )
        self.basic_stats_val = basic_statistics.BasicStatistics(
            X=split.X.val, name='#00b5eb'
        )
        self.class_stats = class_statistics.ClassStatistics(
            X=split.X.train, y=split.y.train
        )
        self.basic_cor_train = basic_correlation.BasicCorrelation(
            X=split.X.train, name='#00b5eb'
        )
        self.basic_cor_test = basic_correlation.BasicCorrelation(
            X=split.X.test, name='#00b5eb'
        )
        self.basic_cor_val = basic_correlation.BasicCorrelation(
            X=split.X.val, name='#00b5eb'
        )
        self.class_cor = class_correlation.ClassCorrelation(
            X=split.X.train, y=split.y.train
        )
        self.pplot = pairplot.PairPlot(X=split.X.train, y=split.y.train)
        self.dis = distributions.Distributions(y=split.y.train)
        self.out = outliers.Outliers(X=split.X.train)
        self.nov_test = novelty.Novelty(
            X_train=split.X.train, X_test=split.X.test
        )
        self.nov_val = novelty.Novelty(
            X_train=split.X.train, X_test=split.X.val
        )

    def __repr__(self) -> str:
        """
        Provide a string representation of the ExploreData object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the ExploreData object.

        Returns
        -------
        str
            A string representation of the ExploreData object.

        Examples
        --------
        >>> print(explore)
        """
        return "ExploreData()"

    def _register_callbacks(self, app: Dash, layouts) -> None:
        """
        Register the callbacks for the Dash app.

        Name
        ----
        _register_callbacks

        Description
        -----------
        Register the callbacks for the Dash app.

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
        layouts : dict[str, html.Div]
            The layouts for the tabs.

        Examples
        --------
        >>> app = Dash(__name__)
        >>> explore._register_callbacks(app)
        """
        app.callback(
            Output(component_id='tabs-content', component_property='children'),
            [Input(component_id='tabs', component_property='value')],
        )(lambda tab: self._render_content(layouts, tab))

    def _setup_layout(self, app: Dash) -> None:
        """
        Set up the layout for the Dash app.

        Name
        ----
        _setup_layout

        Description
        -----------
        Setup the layout for the Dash app.

        Dependencies
        ------------
            External
            --------
            dash
                Package for creating reactive web applications.
            dash_bootstrap_components
                Package for creating Bootstrap components.

        Parameters
        ----------
        app : Dash
            The Dash app.

        Examples
        --------
        >>> app = Dash(__name__)
        >>> app = explore._setup_layout(app)
        """
        app.layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    children=dbc.Col(
                        children=dcc.Tabs(
                            id="tabs",
                            value='dis',
                            children=[
                                dcc.Tab(
                                    label='Data and Generators',
                                    value='gen',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Statistics (Training)',
                                    value='basic_stats_train',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Statistics (Testing)',
                                    value='basic_stats_test',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Statistics (Validation)',
                                    value='basic_stats_val',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='By Class Statistics',
                                    value='class_stats',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Correlation Matrix (Training)',
                                    value='basic_cor_train',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Correlation Matrix (Testing)',
                                    value='basic_cor_test',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Correlation Matrix (Validation)',
                                    value='basic_cor_val',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='By Class Correlation Matrices',
                                    value='class_cor',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Feature Pair Plot',
                                    value='pplot',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                    disabled_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Class Distributions',
                                    value='dis',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                    disabled_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Outliers (Training)',
                                    value='out',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                    disabled_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Novelties (Testing)',
                                    value='nov_test',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                    disabled_style={'color': 'black'},
                                ),
                                dcc.Tab(
                                    label='Novelties (Validation)',
                                    value='nov_val',
                                    style={'color': 'black'},
                                    selected_style={'color': 'black'},
                                    disabled_style={'color': 'black'},
                                ),
                            ],
                        ),
                        width=12,
                    )
                ),
                dbc.Row(
                    children=dbc.Col(
                        children=html.Div(id='tabs-content'), width=12
                    )
                ),
            ],
        )

    def _render_content(
        self, layouts: dict[str, html.Div], tab: str
    ) -> html.Div:
        """
        Render the content based on the selected tab.

        Name
        ----
        _render_content

        Description
        -----------
        Render the content based on the selected tab.

        Dependencies
        ------------
            External
            --------
            dash
                Package for making reactive web applications.

        Parameters
        ----------
        layouts : dict[str, html.Div]
            The layouts for the tabs.
        tab : str
            The selected tab.

        Returns
        -------
        html.Div
            The HTML div containing the content for the selected tab.

        Examples
        --------
        >>> explore._render_content(tab='dis')
        """
        return layouts.get(tab, "Tab not found")

    def __call__(self) -> None:
        """
        Display in a Dash app or Jupyter notebook.

        Name
        ----
        __call__

        Description
        -----------
        Display in a Dash app or Jupyter notebook.

        Dependencies
        ------------
            External
            --------
            dash
                Package for making reactive web applications.
            dash_bootstrap_components
                Package for creating Bootstrap components.

        Examples
        --------
        >>> gen = generate_data.GenerateData()
        >>> split = split_data.SplitData(gen=gen)
        >>> explore = ExploreData(split=split)
        >>> explore()
        """
        layouts = {
            'gen': self.gen('none'),
            'basic_stats_train': self.basic_stats_train('none'),
            'basic_stats_test': self.basic_stats_test('none'),
            'basic_stats_val': self.basic_stats_val('none'),
            'class_stats': self.class_stats('none'),
            'basic_cor_train': self.basic_cor_train('none'),
            'basic_cor_test': self.basic_cor_test('none'),
            'basic_cor_val': self.basic_cor_val('none'),
            'class_cor': self.class_cor('none'),
            'pplot': self.pplot('none'),
            'dis': self.dis('none'),
            'out': self.out('none'),
            'nov_test': self.nov_test('none'),
            'nov_val': self.nov_val('none'),
        }

        app = dash.Dash(
            name=__name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )

        # Setup layout and register callbacks
        self._setup_layout(app=app)
        self._register_callbacks(app=app, layouts=layouts)
        self.class_stats._register_callbacks(app=app)
        self.class_cor._register_callbacks(app=app)

        # Start the server
        app.run_server(debug=True, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    gen = generate_data.GenerateData()
    split = split_data.SplitData(gen=gen)
    explore = ExploreData(split=split)
    explore()
