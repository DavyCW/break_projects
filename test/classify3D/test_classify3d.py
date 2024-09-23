"""
Test file for the classify3d.py file containing 16 tests.

Name
----
test_classify3d.py

Description
-----------
This test suite verifies the functionality of the `GenerateData` class defined
in `classify3d.py`. It covers various aspects of the class, including:

* Initialization with default and custom parameters
* String representation format
* File handling for valid and invalid data
* Sphere face and point generation with different resolutions and centers
* Generation of unique colors and unbalanced label amounts
* Output shapes and label counts for generated data
* Plot data generation for visualization

Dependencies
------------
    External
    --------
    pathlib
        Package for getting the type of the default tmp_path fixture.
    pytest
        Package for creating fixtures.
    numpy
        Package for fast array manipulation.

    Internal
    --------
    GenerateData
        Generates 3D data based on passed file.

Attributes
----------
    Fixtures
    --------
    common_data_file()
        Create a temporary file with common data content.

    Tests
    -----
    test_init_default_params()
        Tests that the __init__ method initializes attributes with defaults.
    test_init_custom_params()
        Tests that the __init__ method uses provided parameters.
    test_repr_string_format()
        Tests if __repr__ returns a string in the correct format.
    test_open_file_valid_data()
        Test _open_file with valid input data.
    test_open_file_invalid_data()
        Test _open_file with invalid input data.
    test_generate_sphere_faces_resolution_10()
        Test _generate_sphere_faces with resolution 10.
    test_generate_sphere_faces_resolution_20()
        Test _generate_sphere_faces with resolution 20.
    test_generate_sphere_points_center_zero()
        Test _generate_sphere_points with center at origin.
    test_generate_sphere_points_shifted_center()
        Test _generate_sphere_points with a shifted center.
    test_generate_unique_colors_count()
        Tests if _generate_unique_colors generates the correct color amount.
    test_generate_unique_colors_uniqueness()
        Tests if _generate_unique_colors generates unique colors.
    test_generate_dirichlet_unbalanced_sum()
        Tests if _generate_dirichlet_unbalanced generates correct amounts.
    test_generate_dirichlet_unbalanced_minimum()
        Tests if _generate_dirichlet_unbalanced ensures minimum values.
    test_generate_data_output_shapes()
        Tests if _generate_data generates the correct output shapes.
    test_generate_data_label_counts()
        Tests if _generate_data generates the correct number of labels.
    test_call_plot_data()
        Tests if __call__ generates the expected plot data.
"""

from pathlib import Path
import pytest
import numpy as np
from src.classify3D.classify3d import GenerateData


@pytest.fixture
def common_data_file(tmp_path: Path) -> Path:
    """
    Create a temporary file with common data content.

    Name
    ----
    common_data_file

    Description
    -----------
    Contains a basic file akin to the types of files that GenerateData is used
    to reading, but with a significantly smaller and simpler amount of data for
    test purposes.

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

    Parameters
    ----------
    tmp_path : Path
        Fixture that creates and cleans up a temporary path.

    Returns
    -------
    Path
        Temporary path with data attached to it.
    """
    content = """1.0\n0\t0.5\t0.5\t0.5\n1\t1.0\t1.0\t1.0\n"""
    file_path = tmp_path / "data.txt"
    file_path.write_text(content)
    return file_path


def test_init_default_params() -> None:
    """
    Tests that the __init__ method initializes attributes with defaults.

    Name
    ----
    test_init_default_params

    Dependencies
    ------------
        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.
    """
    data_generator = GenerateData()

    assert data_generator.file_name == (
        "/workspaces/Break_Projects/data/ssp256.txt"
    )
    assert data_generator.resolution == 30
    assert data_generator.avg_points_per_sphere == 1000
    assert data_generator.min_points_per_sphere == 100
    assert data_generator.radial_scale == 1.0


def test_init_custom_params(common_data_file: Path) -> None:
    """
    Tests that the __init__ method uses provided parameters.

    Name
    ----
    test_init_custom_params

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    data_generator = GenerateData(
        file_name=str(common_data_file),
        resolution=50,
        avg_points_per_sphere=500,
        min_points_per_sphere=50,
        radial_scale=0.8,
    )

    assert data_generator.file_name == str(common_data_file)
    assert data_generator.resolution == 50
    assert data_generator.avg_points_per_sphere == 500
    assert data_generator.min_points_per_sphere == 50
    assert data_generator.radial_scale == 0.8


def test_repr_string_format(common_data_file: Path) -> None:
    """
    Tests if __repr__ returns a string in the correct format.

    Name
    ----
    test_repr_string_format

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    data_generator = GenerateData(
        file_name=str(object=common_data_file), resolution=20
    )
    repr_string = repr(data_generator)

    # Expected format
    expected_format = (
        f"GenerateData(file_name: {common_data_file}, resolution: 20, "
        "avg_points_per_sphere: 1000, min_points_per_sphere: 100, "
        "radial_scale: 1.0)"
    )

    # Assert that the repr string matches the expected format
    assert repr_string == expected_format


def test_open_file_valid_data(common_data_file: Path) -> None:
    """
    Test _open_file with valid input data.

    Name
    ----
    test_open_file_valid_data

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(file_name=str(object=common_data_file))

    # Call the _open_file method
    coords, radius = data_generator._open_file()

    # Expected results
    expected_coords = [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    expected_radius = 1.0

    # Assert the results
    assert coords == expected_coords
    assert radius == expected_radius


def test_open_file_invalid_data(tmp_path: Path) -> None:
    """
    Test _open_file with invalid input data.

    Name
    ----
    test_open_file_invalid_data

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    tmp_path : Path
        Temporary path with no data attached to it.
    """
    # Mock file content with missing radius
    content = """0\t0.5\t0.5\t0.5\n1\t1.0\t1.0\t1.0\n"""

    # Create a temporary file in the provided directory
    temp_file = tmp_path / "data.txt"
    temp_file.write_text(content)

    # Create the GenerateData object
    with pytest.raises(expected_exception=ValueError):
        GenerateData(file_name=str(temp_file))


def test_generate_sphere_faces_resolution_10(common_data_file: Path) -> None:
    """
    Test _generate_sphere_faces with resolution 10.

    Name
    ----
    test_generate_sphere_faces_resolution_10

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(
        file_name=str(common_data_file), resolution=10
    )
    faces = data_generator._generate_sphere_faces()

    # Expected number of faces for resolution 10
    expected_num_faces = 162

    assert faces.shape[0] == expected_num_faces


def test_generate_sphere_faces_resolution_20(common_data_file: Path) -> None:
    """
    Test _generate_sphere_faces with resolution 20.

    Name
    ----
    test_generate_sphere_faces_resolution_20

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(
        file_name=str(common_data_file), resolution=20
    )
    faces = data_generator._generate_sphere_faces()

    # Expected number of faces for resolution 10
    expected_num_faces = 722

    assert faces.shape[0] == expected_num_faces


def test_generate_sphere_points_center_zero(common_data_file: Path) -> None:
    """
    Test _generate_sphere_points with center at origin.

    Name
    ----
    test_generate_sphere_points_center_zero

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(file_name=str(common_data_file))

    center = np.array(object=[0, 0, 0])

    points = data_generator._generate_sphere_points(center=center)

    # Check that all points have a distance close to the radius (1.0)
    assert np.allclose(a=np.linalg.norm(x=points, axis=1), b=1.0, atol=1)


def test_generate_sphere_points_shifted_center(common_data_file: Path) -> None:
    """
    Test _generate_sphere_points with a shifted center.

    Name
    ----
    test_generate_sphere_points_shifted_center

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(file_name=str(common_data_file))
    center = np.array(object=[1, 2, 3])
    points = data_generator._generate_sphere_points(center=center)

    # Check that all points shifted by the center value
    assert np.allclose(a=points[:, 0] - center[0], b=0.0, atol=1)
    assert np.allclose(a=points[:, 1] - center[1], b=0.0, atol=1)
    assert np.allclose(a=points[:, 2] - center[2], b=0.0, atol=1)


def test_generate_unique_colors_count(common_data_file: Path) -> None:
    """
    Tests if _generate_unique_colors generates the correct color amount.

    Name
    ----
    test_generate_unique_colors_count

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(file_name=str(common_data_file))
    colors = data_generator._generate_unique_colors()

    # Assert that the number of colors generated matches the number of spheres
    assert len(colors) == 2


def test_generate_unique_colors_uniqueness(common_data_file: Path) -> None:
    """
    Tests if _generate_unique_colors generates unique colors.

    Name
    ----
    test_generate_unique_colors_uniqueness

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(file_name=str(common_data_file))
    colors = data_generator._generate_unique_colors()

    # Assert that all colors are unique
    assert len(set(colors)) == len(colors)


def test_generate_dirichlet_unbalanced_sum(common_data_file: Path) -> None:
    """
    Tests if _generate_dirichlet_unbalanced generates correct amounts.

    Name
    ----
    test_generate_dirichlet_unbalanced_sum

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(
        file_name=str(common_data_file), avg_points_per_sphere=100
    )
    amounts = data_generator._generate_dirichlet_unbalanced()

    # Assert that the sum of amounts equals the expected total
    expected_total = data_generator.n * data_generator.avg_points_per_sphere
    assert sum(amounts) == expected_total


def test_generate_dirichlet_unbalanced_minimum(common_data_file: Path) -> None:
    """
    Tests if _generate_dirichlet_unbalanced ensures minimum values.

    Name
    ----
    test_generate_dirichlet_unbalanced_minimum

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(
        file_name=str(common_data_file), min_points_per_sphere=20
    )
    amounts = data_generator._generate_dirichlet_unbalanced()

    # Assert that all amounts are greater than or equal to the minimum
    assert all(
        amount >= data_generator.min_points_per_sphere for amount in amounts
    )


def test_generate_data_output_shapes(common_data_file: Path) -> None:
    """
    Tests if _generate_data generates the correct output shapes.

    Name
    ----
    test_generate_data_output_shapes

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    dg = GenerateData(file_name=str(common_data_file))
    plot_data, X, y = dg._generate_data()

    # Assert the expected shapes
    expected_num_points = dg.n * dg.avg_points_per_sphere
    assert len(plot_data) == 2 * dg.n  # 2 plots per sphere
    assert X.shape == (expected_num_points, 6)
    assert y.shape == (expected_num_points,)


def test_generate_data_label_counts(common_data_file: Path) -> None:
    """
    Tests if _generate_data generates the correct number of labels.

    Name
    ----
    test_generate_data_label_counts

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(
        file_name=str(common_data_file), avg_points_per_sphere=100
    )
    _, _, y = data_generator._generate_data()

    # Assert that the number of labels for each class matches the expected
    unique_labels, label_counts = np.unique(ar=y, return_counts=True)
    expected_amounts = data_generator._generate_dirichlet_unbalanced()
    assert np.allclose(a=label_counts, b=expected_amounts)


def test_call_plot_data(common_data_file: Path) -> None:
    """
    Tests if __call__ generates the expected plot data.

    Name
    ----
    test_call_plot_data

    Dependencies
    ------------
        External
        --------
        pathlib
            Package for getting the type of the default tmp_path fixture.
        pytest
            Package for creating fixtures.

        Internal
        --------
        GenerateData
            Generates 3D data based on passed file.

    Parameters
    ----------
    common_data_file : Path
        Temporary path with data attached to it.
    """
    # Create the GenerateData object with the temporary file path
    data_generator = GenerateData(file_name=str(common_data_file))
    data_generator('blank')  # Call the instance to generate the plot

    # Assert that the plot data contains the expected number of traces (2)
    assert len(data_generator.plot_data) == 4  # 2 spheres, each with 2 traces
