"""
Module containing the GenerateData class.

Name
----
generate_data.py

Description
-----------
Contains a class to take a given packing of spheres inside the unit sphere and
generate points inside each sphere uniformly for a classification problem.
Plots the points and their spheres using plotly, using a unique color for each
sphere which also acts as that sphere's label.

Dependencies
------------
    External
    --------
    numpy
        Package for fast array manipulation.
    scipy
        Package for statistical distributions.
    typing
        Package for more descriptive type hints.
    matplotlib
        Package for good color distributions.
    plotly
        Package to plot 3D data beautifully.
    dash
        Package for making reactive web applications.

Attributes
----------
    Classes
    -------
    GenerateData
        Generates 3D data based on passed file.
"""

import numpy as np
from scipy.stats import uniform, dirichlet
from typing import List
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import plotly.graph_objects as go
import dash
from dash import dcc, html


class GenerateData:
    """
    Generates 3D data based on passed file.

    Name
    ----
    GenerateData

    Description
    -----------
    A class to generate and manipulate 3D data points within and on the surface
    of spheres based on input parameters and specifications. This class reads
    coordinates and radius from a given file and uses various methods to
    generate points, faces, and visualizations of the data.

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.
        scipy
            Package for statistical distributions.
        typing
            Package for more descriptive type hints.
        matplotlib
            Package for good color distributions.
        plotly
            Package to plot 3D data beautifully.
        dash
            Package for making reactive web applications.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initializes the object with given parameters, reads data from a
            file, and generates initial data points.
        __repr__()
            Provide a string representation of the GenerateData object.
        _open_file()
            Reads the file and extracts coordinates and radius values.
        _generate_sphere_faces()
            Generates the faces of the sphere as triangles for visualization.
        _generate_sphere_points()
            Generates points on the surface of a sphere based on the given
            resolution.
        _generate_points_in_sphere()
            Generates random points uniformly distributed within a sphere.
        _generate_unique_colors()
            Generates a list of unique colors for plotting different spheres.
        _generate_dirichlet_unbalanced()
            Generates unbalanced class distributions using a Dirichlet
            distribution.
        _generate_data()
            Generates all data points and organizes them for visualization.
        __call__()
            Plots the generated data in a 3D visualization.

        Variables
        ---------
        file_name : str
            Path to the input file containing initial coordinates and radius
            values.
        coords : np.ndarray
            A 2D array where each row represents the coordinates (x, y, z) of a
            sphere center.
        radius : float
            The radius of the spheres read from the input file.
        resolution : int, optional
            Determines the granularity of the sphere's surface. Higher values
            lead to more detailed spheres. Default is 30.
        avg_points_per_sphere : int, optional
            Average number of points to generate inside each sphere. Default is
            1000.
        min_points_per_sphere : int, optional
            Minimum number of points to generate inside each sphere, ensuring
            that all spheres have a baseline number of points. Default is 100.
        radial_scale : float, optional
            A scaling factor for the radial distance that adjusts the
            distribution of points within the sphere. A value of 1 maintains
            uniformity, while values less than 1 compress the points toward the
            center. Default is 1.0.
        n : int
            Number of spheres represented by the input data.
        plot_data : list
            A list containing the coordinates and attributes for plotting each
            sphere.
        X : np.ndarray
            A 2D array containing the coordinates of all generated points in
            the dataset.
        y : np.ndarray
            A 1D array containing class labels for the generated points.
    """

    def __init__(
        self,
        file_name: str = "/workspaces/Break_Projects/data/ssp256.txt",
        resolution: int = 30,
        avg_points_per_sphere: int = 1000,
        min_points_per_sphere: int = 100,
        radial_scale: float = 1.0,
    ) -> None:
        """
        Initialize the GenerateData object.

        Name
        ----
        __init__

        Description
        -----------
        This constructor reads the input file to obtain initial coordinates and
        radius values, and sets various parameters that control the resolution,
        number of points per sphere, and the distribution of points. The
        constructor also calls the `_generate_data` method to create data
        points based on the specified attributes.

        Dependencies
        ------------
            Internal
            --------
            _open_file()
                Reads the file and extracts coordinates and radius values.
            _generate_data()
                Generates all data points and organizes them for visualization.

        Parameters
        ----------
        file_name : str, optional
            Path to the input file containing initial sphere data, by default
            "/workspaces/Break_Projects/data/ssp256.txt".
        resolution : int, optional
            Granularity of the sphere's surface, by default 30.
        avg_points_per_sphere : int, optional
            Average number of points to generate per sphere, by default 1000.
        min_points_per_sphere : int, optional
            Minimum number of points to generate per sphere, by default 100.
        radial_scale : float, optional
            Scaling factor for radial distance distribution, by default 1.0.

        Examples
        --------
        >>> data_generator = GenerateData()

        >>> data_generator()
        # Plots the generated 3D data using Plotly.
        """
        self.file_name = file_name
        self.coords, self.radius = self._open_file()
        self.resolution = resolution
        self.avg_points_per_sphere = avg_points_per_sphere
        self.min_points_per_sphere = min_points_per_sphere
        self.radial_scale = radial_scale
        self.n = len(self.coords)
        self.plot_data, self.X, self.y = self._generate_data()

    def __repr__(self) -> str:
        """
        Provide a string representation of the GenerateData object.

        Name
        ----
        __repr__

        Description
        -----------
        This method returns a concise summary of the key attributes of the
        GenerateData object, including the file name, resolution, average
        points per sphere, minimum points per sphere, and radial scale.
        This helps to quickly understand the configuration of the object
        during debugging or logging.

        Returns
        -------
        str
            A string summarizing the key parameters of the object.

        Examples
        --------
        >>> data_generator = GenerateData()
        >>> print(data_generator)
        GenerateData(file_name: /workspaces/Break_Projects/data/ssp256.txt,
                     resolution: 30,
                     avg_points_per_sphere: 1000,
                     min_points_per_sphere: 100,
                     radial_scale: 1.0
                     )
        """
        return (
            f"GenerateData("
            f"file_name: {self.file_name}, "
            f"resolution: {self.resolution}, "
            f"avg_points_per_sphere: {self.avg_points_per_sphere}, "
            f"min_points_per_sphere: {self.min_points_per_sphere}, "
            f"radial_scale: {self.radial_scale})"
        )

    def _open_file(self) -> tuple[list[list[float]], float]:
        """
        Read the specified file to extract sphere coordinates and radius.

        Name
        ----
        _open_file

        Description
        -----------
        This method reads the input file specified by `file_name` and extracts
        the radius and the coordinates of the sphere centers. The file format
        is expected to have the radius as the first line, followed by lines of
        space-separated values representing the index and coordinates (x, y, z)
        of each sphere center.

        Returns
        -------
        tuple[list[list[float]], float]
            A tuple containing:
            - A list of lists where each inner list represents the coordinates
            [x, y, z] of a sphere center.
            - A float representing the radius of the spheres.

        Examples
        --------
        Suppose the file contains the following data:

        1.0
        0 0.5 0.5 0.5
        1 1.0 1.0 1.0

        >>> data_generator = GenerateData(file_name="sample_data.txt")
        >>> coords, radius = data_generator._open_file()
        >>> print(coords)
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
        >>> print(radius)
        1.0
        """
        with open(file=self.file_name, mode="r") as f:
            split: list[str] = f.read().split(sep='\n')
            radius: float = float(split[0].strip())
            coordset: list[str] = split[1:-1]
            coords: list[list[float]] = [
                [float(x), float(y), float(z)]
                for _, x, y, z in [c.split() for c in coordset]
            ]
        return coords, radius

    def _generate_sphere_faces(self) -> np.ndarray:
        """
        Generate the faces of a sphere as triangles.

        Name
        ----
        _generate_sphere_faces

        Description
        -----------
        This function generates a set of triangular faces that approximate a
        sphere. The faces are computed based on a grid of specified resolution,
        allowing for a smoother or coarser representation of the sphere.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

        Returns
        -------
        np.ndarray
            An array of shape (N, 3) where N is the number of triangular faces.
            Each row represents the indices of the vertices that form a
            triangle.

        Examples
        --------
        Create a `GenerateData` instance with a specified resolution and use
        the `_generate_sphere_faces` method to generate the faces for a sphere:

        >>> gen = GenerateData(file_name="/path/to/file.txt", resolution=10)
        >>> faces = gen._generate_sphere_faces()
        >>> faces.shape
        (180, 3)
        """
        # Create indices for a grid of shape (resolution - 1, resolution - 1)
        i, j = np.indices(
            dimensions=(self.resolution - 1, self.resolution - 1)
        )

        # Calculate the vertex indices for each corner of the quad
        top_left = i * self.resolution + j
        top_right = top_left + 1
        bottom_left = top_left + self.resolution
        bottom_right = bottom_left + 1

        # Create two triangles for each quad
        faces = np.vstack(
            tup=(
                np.column_stack(
                    tup=(
                        top_left.ravel(),
                        top_right.ravel(),
                        bottom_left.ravel(),
                    )
                ),
                np.column_stack(
                    tup=(
                        top_right.ravel(),
                        bottom_right.ravel(),
                        bottom_left.ravel(),
                    )
                ),
            )
        )

        return faces

    def _generate_sphere_points(self, center: np.ndarray) -> np.ndarray:
        """
        Generate points on the surface of a sphere.

        Name
        ----
        _generate_sphere_points

        Description
        -----------
        This function generates a set of points uniformly distributed on the
        surface of a sphere. The points are computed based on spherical
        coordinates and then converted to Cartesian coordinates, offset by a
        specified center.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.

        Parameters
        ----------
        center : np.ndarray
            The center of the sphere as a 1D array of shape (3,), representing
            the coordinates in 3D space.

        Returns
        -------
        np.ndarray
            An array of shape (N, 3) where N is the total number of points
            generated on the sphere. Each row represents the (x, y, z)
            coordinates of a point.

        Examples
        --------
        Create a `GenerateData` instance and use the `_generate_sphere_points`
        method to generate points on a sphere centered at the specified
        coordinates:

        >>> gen = GenerateData(file_name="/path/to/file.txt", resolution=10)
        >>> center = np.array([1.0, 2.0, 3.0])
        >>> points = gen._generate_sphere_points(center=center)
        >>> points.shape
        (100, 3)
        """
        # Create a grid of theta (0 to pi) and phi (0 to 2pi)
        theta, phi = np.meshgrid(
            np.linspace(start=0, stop=np.pi, num=self.resolution),
            np.linspace(start=0, stop=2 * np.pi, num=self.resolution),
        )

        # Calculate Cartesian coordinates for the points on the sphere
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        # Stack the coordinates and add the center offset
        sphere_points = (
            np.column_stack(tup=(x.ravel(), y.ravel(), z.ravel())) + center
        )

        return sphere_points

    def _generate_points_in_sphere(
        self, center: np.ndarray, num_points: int
    ) -> np.ndarray:
        """
        Generate random points uniformly distributed within a sphere.

        Name
        ----
        _generate_points_in_sphere

        Description
        -----------
        This function generates a specified number of random points that are
        uniformly distributed within a sphere of a given radius. The points are
        generated in spherical coordinates and then converted to Cartesian
        coordinates, offset by a specified center.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.
            scipy
                Package for statistical distributions.

        Parameters
        ----------
        center : np.ndarray
            The center of the sphere as a 1D array of shape (3,), representing
            the coordinates in 3D space.
        num_points : int
            The total number of random points to generate within the sphere.

        Returns
        -------
        np.ndarray
            An array of shape (N, 6) where N is the number of points generated.
            Each row represents the (x, y, z, r theta phi) coordinates.

        Examples
        --------
        Create a `GenerateData` instance and use the
        `_generate_points_in_sphere` method to generate points uniformly
        distributed within a sphere centered at the specified coordinates:

        >>> gen = GenerateData(file_name="/path/to/file.txt", resolution=10)
        >>> center = np.array([0.0, 0.0, 0.0])
        >>> num_points = 100
        >>> points = gen._generate_points_in_sphere(center=center,
                                                    num_points=num_points
                                                    )
        >>> points.shape
        (100, 6)
        """
        # Generate random spherical coordinates
        u = uniform.rvs(size=num_points)
        cos_theta = uniform.rvs(loc=-1, scale=2, size=num_points)
        phi = uniform.rvs(scale=2 * np.pi, size=num_points)  # Azimuthal angle

        # Compute spherical coordinates
        r = self.radius * np.cbrt(u) * self.radial_scale  # Radial distance
        theta = np.arccos(cos_theta)  # Polar angle

        # Convert spherical coordinates to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Stack and shift points by center
        points = np.vstack(
            tup=(x + center[0], y + center[1], z + center[2], r, theta, phi)
        ).T

        return points

    def _generate_unique_colors(self) -> List[str]:
        """
        Generate a list of unique colors.

        Name
        ----
        _generate_unique_colors

        Description
        -----------
        This function generates a specified number of unique colors by sampling
        from the 'rainbow' colormap. The colors are returned in hexadecimal
        format and are shuffled to ensure randomness.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.
            typing
                Package for more descriptive type hints.
            matplotlib
                Package for good color distributions.

        Returns
        -------
        List[str]
            A list of unique colors represented as hexadecimal strings.

        Examples
        --------
        Create a `GenerateData` instance and use the `_generate_unique_colors`
        method to generate a list of unique colors:

        >>> gen = GenerateData(file_name="/path/to/file.txt",
                               resolution=10
                               )
        >>> colors = gen._generate_unique_colors()

        Verify that each color is represented as a hexadecimal string:

        >>> all(isinstance(color, str) and
                color.startswith('#') for color in colors
                )
        True
        """
        # Generate a colormap from 'rainbow'
        cmap = plt.get_cmap(name='rainbow')

        # Generate equally spaced points in the colormap
        colors = [cmap(i / (self.n - 1)) for i in range(self.n)]

        # Convert colors to hexadecimal format
        unique_colors = [mcolors.rgb2hex(c=color) for color in colors]

        np.random.shuffle(x=unique_colors)
        return unique_colors

    def _generate_dirichlet_unbalanced(self) -> np.ndarray:
        """
        Generate unbalanced amounts using a Dirichlet distribution.

        Name
        ----
        _generate_dirichlet_unbalanced

        Description
        -----------
        This function generates a set of unbalanced integer amounts that sum to
        a specified total. It uses a Dirichlet distribution to create
        proportions, scales them to the desired total sum, and ensures that
        each amount meets a minimum value requirement.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.
            scipy
                Package for statistical distributions.

        Returns
        -------
        np.ndarray
            An array of integers representing the generated amounts, which sum
            to the specified total and respect the minimum value constraint.

        Examples
        --------
        >>> gen = GenerateData(file_name='data.txt', resolution=10)
        >>> amounts = gen._generate_dirichlet_unbalanced()
        >>> amounts
        array([24, 22, 19, 15, 20])
        >>> np.sum(amounts)
        100
        >>> np.all(amounts >= gen.min_points_per_sphere)
        True
        """
        # Step 1: Generate Dirichlet distributed values
        dirichlet_values = dirichlet.rvs(alpha=np.ones(shape=self.n), size=1)[
            0
        ]

        # Step 2: Scale these values to the target total sum minus the minimum
        scaled_values = (
            dirichlet_values
            * self.n
            * (self.avg_points_per_sphere - self.min_points_per_sphere)
        )

        # Step 3: Convert to integers while ensuring a minimum value
        floored_values = np.floor(scaled_values).astype(int)
        integer_values = floored_values + self.min_points_per_sphere

        # Step 4: Handle rounding errors
        current_sum = np.sum(a=integer_values)
        difference = self.n * self.avg_points_per_sphere - current_sum

        # Get the decimal parts and their indices
        decimal_parts = scaled_values - floored_values
        # Sort indices based on the decimal parts in descending order
        sorted_indices = np.argsort(a=-decimal_parts)

        # Adjust the values with the largest decimal parts first
        for i in range(difference):
            integer_values[sorted_indices[i]] += 1

        return integer_values

    def _generate_data(self) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Plot spheres and random points within them in a 3D space.

        Name
        ----
        _generate_data

        Description
        -----------
        This function generates and visualizes spheres in a 3D plot, along with
        random points distributed within each sphere. The number of points per
        sphere is determined by a Dirichlet distribution to create an
        unbalanced distribution of points. The spheres are rendered with unique
        colors for differentiation.

        Dependencies
        ------------
            External
            --------
            numpy
                Package for fast array manipulation.
            plotly
                Package to plot 3D data beautifully.

            Internal
            --------
                _generate_dirichlet_unbalanced()
                    Generates unbalanced class distributions using a Dirichlet
                    distribution.
                _generate_unique_colors()
                    Generates a list of unique colors for plotting different
                    spheres.
                _generate_sphere_faces()
                    Generates the faces of the sphere as triangles for
                    visualization.
                _generate_sphere_points()
                    Generates random points uniformly distributed within a
                    sphere.
                _generate_points_in_sphere()
                    Generates points on the surface of a sphere based on the
                    given resolution.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - A list of 3D plotly plots.
            - An array of shape (m, 3) with the coordinates of the random
            points, where m is the total number of points generated across all
            spheres.
            - An array of color labels corresponding to each random point.

        Examples
        --------
        >>> gen = GenerateData(file_name='data.txt', resolution=10)
        >>> plot_data, X, y = gen._generate_data()
        >>> len(plot_data)
        4  # Assuming 2 spheres, each with one Mesh3d and one Scatter3d plot.
        >>> X.shape
        (2000, 6)  # Assuming a mean of 1000 points per sphere and 2 spheres.
        >>> y.shape
        (2000, )  # Labels for each of the points in X.
        """
        # Step 1: Generate a list of n unbalanced integers that add up to x.
        dirichlet_unbalanced = self._generate_dirichlet_unbalanced()

        # Step 2: Generate a list of n unique random colors.
        unique_colors = self._generate_unique_colors()

        # Step 3: Initialize X since we'll be concatenating with it.
        X = np.empty(shape=(0, 6), dtype=float)

        # Step 4: Copy the unique labels according to the shape of X.
        y = np.array(
            object=[
                color
                for color, count in zip(unique_colors, dirichlet_unbalanced)
                for _ in range(count)
            ]
        )

        flag = True

        plot_data = []

        sphere_faces = self._generate_sphere_faces()

        for i, coord in enumerate(iterable=self.coords):
            center = np.array(object=coord)
            # Generate a sphere.
            sphere_points = self._generate_sphere_points(center=center)
            plot_data.append(
                go.Mesh3d(
                    x=sphere_points[:, 0],
                    y=sphere_points[:, 1],
                    z=sphere_points[:, 2],
                    i=sphere_faces[:, 0],
                    j=sphere_faces[:, 1],
                    k=sphere_faces[:, 2],
                    opacity=0.5,
                    color=unique_colors[i],
                    name='Inner Spheres',
                    legendgroup='Inner Spheres',
                    showlegend=flag,
                )
            )

            # Generate random points inside the current sphere.
            random_points = self._generate_points_in_sphere(
                center=center, num_points=dirichlet_unbalanced[i]
            )
            plot_data.append(
                go.Scatter3d(
                    x=random_points[:, 0],
                    y=random_points[:, 1],
                    z=random_points[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=unique_colors[i]),
                    name='Random Points',
                    legendgroup='Random Points',
                    showlegend=flag,
                )
            )

            X = np.concatenate((X, random_points), axis=0)

            flag = False

        return plot_data, X, y

    def __call__(self, display: str = 'notebook') -> None:
        """
        Create and display a 3D plot using Plotly.

        Name
        ----
        __call__

        Description
        -----------
        This function generates a 3D plot based on the provided plot data. It
        configures the layout to hide axes and their tick marks, creating a
        clean visualization environment.

        Dependencies
        ------------
            External
            --------
            plotly
                Package to plot 3D data beautifully.
            dash
                Package for making reactive web applications.

        Parameters
        ----------
        display : str, optional
            Mode for displaying plot, by default 'notebook'

        Examples
        --------
        >>> generate_data = GenerateData(file_name='data.txt', resolution=10)
        >>> generate_data()  # Call the instance to display the plot
        """
        fig = go.Figure(data=self.plot_data)

        fig.update_layout(
            scene_aspectmode='data',
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    title='',
                    showbackground=False,
                    tickvals=[],  # Hide tick values
                    ticktext=[],  # Hide tick text
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    title='',
                    showbackground=False,
                    tickvals=[],  # Hide tick values
                    ticktext=[],  # Hide tick text
                ),
                zaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    title='',
                    showbackground=False,
                    tickvals=[],  # Hide tick values
                    ticktext=[],  # Hide tick text
                ),
            ),
        )

        if display == 'container':
            # Initialize the Dash app
            app = dash.Dash(__name__)
            # Define the layout of the Dash app
            app.layout = html.Div([dcc.Graph(id='3d-plot', figure=fig)])
            app.run_server(debug=True, host='0.0.0.0', port=8050)
        elif display == 'notebook':
            fig.show()


if __name__ == "__main__":
    gen = GenerateData()
    gen('container')
