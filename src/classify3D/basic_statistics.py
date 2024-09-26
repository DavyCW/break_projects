"""
Module calculates and displays basic statistics of a dataset.

Name
----
basic_statistics.py

Description
-----------
This module provides a class for calculating and displaying basic statistics of
a dataset.

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
    BasicStatistics
        A class representing basic statistics of a dataset.
"""

import numpy as np
import pandas as pd
import dash
from dash import html
from dash.dash_table import DataTable
from IPython.display import display as notebook_display
from pandas.io.formats.style import Styler
from src.classify3D.generate_data import GenerateData  # type: ignore
from src.classify3D.split_data import SplitData  # type: ignore
from matplotlib import colors as mcolors
import colorsys
from typing import Optional, Any
import dash_bootstrap_components as dbc


class BasicStatistics:
    """
    A class representing basic statistics of a dataset.

    Name
    ----
    BasicStatistics

    Description
    -----------
    This class provides methods for calculating basic statistics of a dataset,
    such as mean, median, standard deviation, minimum, and maximum.

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
        _get_basic_statistics_dataframe()
            Calculate basic statistics of a dataset.
        _lighten_color()
            Lightens a color by a given amount.
        _style_dataframe()
            Styles the dataframe.
        __call__()
            Display the basic statistics of the dataset.
        _layout_dash()
            Layout the dash app.

        Variables
        ---------
        basic_statistics : pd.DataFrame
            A DataFrame containing the basic statistics of the dataset.
    """

    def __init__(self, X: np.ndarray, name: str) -> None:
        """
        Initialize the BasicStatistics object with a dataset.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the BasicStatistics object with a dataset.

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
        >>> basic_statistics = BasicStatistics(X=np.array([1, 2, 3, 4, 5]),
        ...                                    name='my_dataset'
        ...                                    )
        """
        self.basic_statistics = self._get_basic_statistics_dataframe(
            X=X, name=name
        )
        self.header_color = self._lighten_color(
            color=self.basic_statistics.name, amount=0.3
        )
        self.band_color = self._lighten_color(
            color=self.basic_statistics.name, amount=0.15
        )
        self.light_color = self._lighten_color(
            color=self.basic_statistics.name, amount=0.1
        )

    def __repr__(self) -> str:
        """
        Provide a string representation of the BasicStatistics object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the BasicStatistics object.

        Returns
        -------
        str
            A string representation of the BasicStatistics object.

        Examples
        --------
        >>> print(basic_statistics)
        BasicStatistics(basic_statistics='my_dataset')
        """
        return (
            f"BasicStatistics(basic_statistics={self.basic_statistics.name})"
        )

    def _get_basic_statistics_dataframe(
        self, X: np.ndarray, name: str
    ) -> pd.DataFrame:
        """
        Calculate basic statistics of a dataset.

        Name
        ----
        _get_basic_statistics_dataframe

        Description
        -----------
        Calculate basic statistics of a dataset.

        Dependencies
        ------------
            External
            --------
            numpy
                A library for efficient numerical computation.
            pandas
                Package for data manipulation and analysis.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        name : str
            The name of the dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the basic statistics of the dataset.

        Examples
        --------
        >>> basic_statistics = BasicStatistics._get_basic_statistics_dataframe(
        ...                             X=np.array([1, 2, 3, 4, 5]),
        ...                             name='my_dataset'
        ...                             )
        """
        mean = np.mean(a=X, axis=0)
        median = np.median(a=X, axis=0)
        std = np.std(a=X, axis=0)
        minimum = np.min(a=X, axis=0)
        maximum = np.max(a=X, axis=0)
        row_labels = ['x', 'y', 'z', 'r', 'theta', 'phi']
        df = pd.DataFrame(
            data={
                'mean': mean,
                'median': median,
                'std': std,
                'minimum': minimum,
                'maximum': maximum,
            },
            index=row_labels,
        )
        df.name = name
        return df

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

    def _style_dataframe(self, df: pd.DataFrame) -> Styler:
        """
        Display the basic statistics of the dataset.

        Name
        ----
        _style_dataframe

        Description
        -----------
        Display the basic statistics of the dataset.

        Dependencies
        ------------
            External
            --------
            pandas
                Package for data manipulation and analysis.

        Parameters
        ----------
        df : pd.DataFrame
            The basic statistics of the dataset.

        Returns
        -------
        Styler
            The styled DataFrame.

        Examples
        --------
        >>> basic_statistics = BasicStatistics._style_dataframe(
        ...                             df=pd.DataFrame(data={'x': [1, 2, 3],
        ...                                                    'y': [4, 5, 6]
        ...                                                    })
        ...                             )
        """
        # Define the styles for the header and index
        styles = {
            'selector': 'thead th',
            'props': [
                ('background-color', self.header_color),
                ('color', '#000000'),
                ('font-weight', 'bold'),
                ('fontFamily', 'Arial'),
                ('text-align', 'center'),
                ('padding', '10px'),
            ],
        }
        index_styles = {
            'selector': 'th.row_heading',
            'props': [
                ('background-color', self.header_color),
                ('color', '#000000'),
                ('font-weight', 'bold'),
                ('fontFamily', 'Arial'),
                ('text-align', 'center'),
                ('padding', '10px'),
            ],
        }

        # Apply the styles
        styled_df = df.style.set_table_styles(
            table_styles=[styles, index_styles]
        )

        # Alternate row colors
        styled_df = styled_df.apply(
            lambda x: [
                (
                    f'background-color: {self.band_color};'
                    if i % 2 == 0
                    else f'background-color: {self.light_color};'
                )
                for i in range(len(x))
            ],
            axis=0,
        )

        styled_df = styled_df.format(precision=3)

        # Center the text and add 10px padding on all sides
        styled_df = styled_df.set_properties(
            **{
                'text-align': 'center',
                'padding': '10px',
                'color': '#000000',
                'fontFamily': 'Arial',
                'margin': 'auto',
                'border': '1px solid #ddd',
            }
        )

        return styled_df

    def __call__(self, display: str = 'notebook') -> Optional[html.Div]:
        """
        Display the basic statistics of the dataset.

        Name
        ----
        __call__

        Description
        -----------
        Display the basic statistics of the dataset in a Jupyter notebook or as
        a Dash app.

        Dependencies
        ------------
            External
            --------
            dash
                Package for making reactive web applications.
            ipykernel
                Package for displaying plots in Jupyter notebooks.
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
        >>> basic_statistics = BasicStatistics(X=np.array([1, 2, 3, 4, 5]),
        ...                                    name='my_dataset'
        ...                                    )
        >>> basic_statistics()
        >>> basic_statistics(display='container')
        """
        if display == 'notebook':
            # Display the DataFrame within a Jupyter notebook
            notebook_display(self._style_dataframe(df=self.basic_statistics))

        elif display == 'container' or display == 'none':
            # Include the index as a column and reset the DataFrame
            row_df = self.basic_statistics.reset_index().rename(
                columns={'index': 'Feature'}
            )

            # Format data to 3 decimal points
            data = row_df.round(decimals=3).to_dict(orient='records')

            app = dash.Dash(
                name=__name__, external_stylesheets=[dbc.themes.DARKLY]
            )

            app.layout = self._layout_dash(row_df=row_df, data=data)

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

    def _layout_dash(
        self, row_df: pd.DataFrame, data: list[dict[str, Any]]
    ) -> dbc.Container:
        """
        Layout the Dash app.

        Name
        ----
        _layout_dash

        Description
        -----------
        Layout the Dash app.

        Dependencies
        ------------
            External
            --------
            pandas
                Package for data manipulation and analysis.
            dash
                Package for making reactive web applications.
            dash_bootstrap_components
                Package for creating Bootstrap components.

        Parameters
        ----------
        row_df : pd.DataFrame
            Dataframe for the table columns.
        data : list[dict[str, Any]]
            Data for the table.

        Returns
        -------
        dbc.Container
            Dash layout for the table.

        Examples
        --------
        >>> row_df = pd.DataFrame({'Feature': ['mean',
        ...                                    'median',
        ...                                    'std',
        ...                                    'min',
        ...                                    'max'
        ...                                    ],
        ...                        'Value': [3.0,
        ...                                  3.0,
        ...                                  1.5811388300841898,
        ...                                  1.0,
        ...                                  5.0
        ...                                  ]
        ...                        })
        >>> data = [{'Feature': 'mean', 'Value': 3.0},
        ...         {'Feature': 'median', 'Value': 3.0},
        ...         {'Feature': 'std', 'Value': 1.5811388300841898},
        ...         {'Feature': 'min', 'Value': 1.0},
        ...         {'Feature': 'max', 'Value': 5.0}]
        >>> layout = BasicStatistics._layout_dash(row_df=row_df, data=data)
        >>> layout
        dbc.Container(children=[
        ... html.H1(children='Basic Statistics',
        ...         style={'textAlign': 'center',
        ...                'color': 'black',
        ...                'fontFamily': 'Arial',
        ...                'fontSize': '30px',
        ...                'width': '60%',
        ...                'margin': 'auto',
        ...                'padding-top': '20px'
        ...                }
        ...         ),
        ... dbc.Table.from_dataframe(row_df,
        ...                          striped=True,
        ...                          bordered=True,
        ...                          hover=True
        ...                          )
        ... ])
        """
        return dbc.Container(
            children=[
                html.H1(
                    children=self.basic_statistics.name,
                    style={
                        'textAlign': 'center',
                        'color': self.basic_statistics.name,
                        'fontFamily': 'Arial',
                        'fontSize': '30px',
                        'width': '60%',
                        'margin': 'auto',
                        'padding-top': '20px',
                        'padding-bottom': '20px',
                    },
                ),
                DataTable(
                    columns=[{'name': i, 'id': i} for i in row_df.columns],
                    data=data,
                    style_table={
                        'width': '80%',
                        'margin': 'auto',
                        'border': '2px solid #ccc',
                    },
                    style_cell={
                        'padding-top': '10px',
                        'padding-bottom': '10px',
                        'fontSize': '18px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'border': '1px solid #ddd',
                        'backgroundColor': self.band_color,
                    },
                    style_header={
                        'textAlign': 'center',
                        'fontFamily': 'Arial',
                        'backgroundColor': self.header_color,
                        'fontWeight': 'bold',
                        'fontSize': '20px',
                        'borderBottom': '2px solid #ccc',
                        'color': '#000000',
                    },
                    style_header_conditional=[
                        {
                            'if': {'column_id': 'Feature'},
                            'fontSize': '24px',
                        },
                    ],
                    style_data={
                        'textAlign': 'center',
                        'fontFamily': 'Arial',
                        'width': '18%',
                        'color': '#000000',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': self.light_color,
                        },
                        {
                            'if': {'column_id': 'Feature'},
                            'width': '10%',
                            'backgroundColor': self.header_color,
                            'fontWeight': 'bold',
                            'fontSize': '20px',
                            'borderTop': '1px solid #ddd',
                            'borderRight': '2px solid #ccc',
                        },
                    ],
                ),
            ],
            style={},
        )


if __name__ == "__main__":
    gen = GenerateData()
    split = SplitData(gen=gen)
    basic_stats = BasicStatistics(split.X.train, name='#00b5eb')
    basic_stats('container')
