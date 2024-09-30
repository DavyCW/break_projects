"""
Module for testing TransformData class and its methods.

Name
----
test_transform_data.py

Description
-----------
This module contains functions and classes for testing the transform_data
module.

Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        transform_data
            Module provides classes and functions for transforming data.
        split_data
            Module contains functions and classes for splitting data.

Attributes
----------
    Functions
    ---------
    test_transformed_data_initialization()
        Test TransformedData initialization with mocked Data objects.
    test_transformed_data_repr()
        Test TransformedData __repr__ method.
    test_data_transformer_initialization()
        Test DataTransformer initialization.
    test_data_transformer_call()
        Test that __call__ applies each transformation properly.
    test_transform_data_initialization()
        Test the initialization of TransformData with mock SplitData object.
    test_transform_data_repr()
        Test the __repr__ method of TransformData.
"""

from unittest.mock import MagicMock, patch
from src.classify3D.transform_data import (
    DataTransformer,
    TransformedData,
    TransformData,
)
from src.classify3D.split_data import Data


def test_transformed_data_initialization() -> None:
    """
    Test TransformedData initialization with mocked Data objects.

    Name
    ----
    test_transformed_data_initialization

    Description
    -----------
    Test TransformedData initialization with mocked Data objects.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        transform_data
            Module provides classes and functions for transforming data.
    """
    mock_data = MagicMock(spec=Data)
    transformed_data = TransformedData(
        untransformed=mock_data,
        standard_scaled=mock_data,
        power_transformed=mock_data,
        robust_scaled=mock_data,
        min_max_scaled=mock_data,
        quantile_transformed=mock_data,
        max_abs_scaled=mock_data,
    )

    # Assert that attributes were set correctly
    assert transformed_data.untransformed == mock_data
    assert transformed_data.standard_scaled == mock_data
    assert transformed_data.power_transformed == mock_data
    assert transformed_data.robust_scaled == mock_data
    assert transformed_data.min_max_scaled == mock_data
    assert transformed_data.quantile_transformed == mock_data
    assert transformed_data.max_abs_scaled == mock_data


def test_transformed_data_repr() -> None:
    """
    Test TransformedData __repr__ method.

    Name
    ----
    test_transformed_data_repr

    Description
    -----------
    Test TransformedData __repr__ method.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        transform_data
            Module provides classes and functions for transforming data.
        split_data
            Module contains functions and classes for splitting data.
    """
    mock_data = MagicMock(spec=Data)
    transformed_data = TransformedData(
        untransformed=mock_data,
        standard_scaled=mock_data,
        power_transformed=mock_data,
        robust_scaled=mock_data,
        min_max_scaled=mock_data,
        quantile_transformed=mock_data,
        max_abs_scaled=mock_data,
    )

    # Assert __repr__ returns a string and contains expected parts
    result = repr(transformed_data)
    assert isinstance(result, str)
    assert "TransformedData" in result


@patch('src.classify3D.transform_data.StandardScaler')
@patch('src.classify3D.transform_data.PowerTransformer')
@patch('src.classify3D.transform_data.RobustScaler')
@patch('src.classify3D.transform_data.MinMaxScaler')
@patch('src.classify3D.transform_data.QuantileTransformer')
@patch('src.classify3D.transform_data.MaxAbsScaler')
def test_data_transformer_initialization(
    mock_standard: MagicMock,
    mock_power: MagicMock,
    mock_robust: MagicMock,
    mock_min_max: MagicMock,
    mock_quantile: MagicMock,
    mock_max_abs: MagicMock,
) -> None:
    """
    Test DataTransformer initialization.

    Name
    ----
    test_data_transformer_initialization

    Description
    -----------
    Test DataTransformer initialization.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        transform_data
            Module provides classes and functions for transforming data.

    Parameters
    ----------
    mock_standard : MagicMock
        Mock StandardScaler object
    mock_power : MagicMock
        Mock PowerTransformer object
    mock_robust : MagicMock
        Mock RobustScaler object
    mock_min_max : MagicMock
        Mock MinMaxScaler object
    mock_quantile : MagicMock
        Mock QuantileTransformer object
    mock_max_abs : MagicMock
        Mock MaxAbsScaler object
    """
    transformer = DataTransformer()

    # Assert each scaler is initialized properly
    assert isinstance(transformer.standard_scaler, MagicMock)
    assert isinstance(transformer.power_transformer, MagicMock)
    assert isinstance(transformer.robust_scaler, MagicMock)
    assert isinstance(transformer.min_max_scaler, MagicMock)
    assert isinstance(transformer.quantile_transformer, MagicMock)
    assert isinstance(transformer.max_abs_scaler, MagicMock)


@patch('src.classify3D.transform_data.StandardScaler')
@patch('src.classify3D.transform_data.PowerTransformer')
@patch('src.classify3D.transform_data.RobustScaler')
@patch('src.classify3D.transform_data.MinMaxScaler')
@patch('src.classify3D.transform_data.QuantileTransformer')
@patch('src.classify3D.transform_data.MaxAbsScaler')
def test_data_transformer_call(
    mock_max_abs_scaler: MagicMock,
    mock_quantile_transformer: MagicMock,
    mock_min_max_scaler: MagicMock,
    mock_robust_scaler: MagicMock,
    mock_power_transformer: MagicMock,
    mock_standard_scaler: MagicMock,
) -> None:
    """
    Test that __call__ applies each transformation properly.

    Name
    ----
    test_data_transformer_call

    Description
    -----------
    Test that __call__ applies each transformation properly.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        transform_data
            Module provides classes and functions for transforming data.

    Parameters
    ----------
    mock_max_abs_scaler : MagicMock
        Mock object for MaxAbsScaler
    mock_quantile_transformer : MagicMock
        Mock object for QuantileTransformer
    mock_min_max_scaler : MagicMock
        Mock object for MinMaxScaler
    mock_robust_scaler : MagicMock
        Mock object for RobustScaler
    mock_power_transformer : MagicMock
        Mock object for PowerTransformer
    mock_standard_scaler : MagicMock
        Mock object for StandardScaler
    """
    # Create mock objects for the scalers
    mock_standard = mock_standard_scaler.return_value
    mock_power = mock_power_transformer.return_value
    mock_robust = mock_robust_scaler.return_value
    mock_min_max = mock_min_max_scaler.return_value
    mock_quantile = mock_quantile_transformer.return_value
    mock_max_abs = mock_max_abs_scaler.return_value

    # Mock the return value of the `fit_transform` and `transform` methods
    mock_standard.fit_transform.return_value = [[1.0], [2.0]]
    mock_standard.transform.return_value = [[3.0], [4.0]]

    mock_power.fit_transform.return_value = [[5.0], [6.0]]
    mock_power.transform.return_value = [[7.0], [8.0]]

    mock_robust.fit_transform.return_value = [[9.0], [10.0]]
    mock_robust.transform.return_value = [[11.0], [12.0]]

    mock_min_max.fit_transform.return_value = [[13.0], [14.0]]
    mock_min_max.transform.return_value = [[15.0], [16.0]]

    mock_quantile.fit_transform.return_value = [[17.0], [18.0]]
    mock_quantile.transform.return_value = [[19.0], [20.0]]

    mock_max_abs.fit_transform.return_value = [[21.0], [22.0]]
    mock_max_abs.transform.return_value = [[23.0], [24.0]]

    # Mock the Data object and its train, test, and val attributes
    mock_data = MagicMock(spec=Data)
    mock_data.train = [[0.1], [0.2]]
    mock_data.test = [[0.3], [0.4]]
    mock_data.val = [[0.5], [0.6]]

    # Instantiate the DataTransformer and call it with the mock data
    transformer = DataTransformer()
    transformed_data = transformer(X=mock_data)

    # Assert that the result is an instance of TransformedData
    assert isinstance(transformed_data, TransformedData)

    # Verify that each scaler's fit_transform was called with the train data
    mock_standard.fit_transform.assert_called_once_with(X=mock_data.train)
    mock_power.fit_transform.assert_called_once_with(X=mock_data.train)
    mock_robust.fit_transform.assert_called_once_with(X=mock_data.train)
    mock_min_max.fit_transform.assert_called_once_with(X=mock_data.train)
    mock_quantile.fit_transform.assert_called_once_with(X=mock_data.train)
    mock_max_abs.fit_transform.assert_called_once_with(X=mock_data.train)

    # Verify that each scaler's transform was called with test and val data
    mock_standard.transform.assert_any_call(X=mock_data.test)
    mock_standard.transform.assert_any_call(X=mock_data.val)

    mock_power.transform.assert_any_call(X=mock_data.test)
    mock_power.transform.assert_any_call(X=mock_data.val)

    mock_robust.transform.assert_any_call(X=mock_data.test)
    mock_robust.transform.assert_any_call(X=mock_data.val)

    mock_min_max.transform.assert_any_call(X=mock_data.test)
    mock_min_max.transform.assert_any_call(X=mock_data.val)

    mock_quantile.transform.assert_any_call(X=mock_data.test)
    mock_quantile.transform.assert_any_call(X=mock_data.val)

    mock_max_abs.transform.assert_any_call(X=mock_data.test)
    mock_max_abs.transform.assert_any_call(X=mock_data.val)

    # Verify the contents of the transformed data (mocked outputs)
    assert transformed_data.standard_scaled.train == [[1.0], [2.0]]
    assert transformed_data.standard_scaled.test == [[3.0], [4.0]]
    assert transformed_data.standard_scaled.val == [[3.0], [4.0]]

    assert transformed_data.power_transformed.train == [[5.0], [6.0]]
    assert transformed_data.robust_scaled.train == [[9.0], [10.0]]
    assert transformed_data.min_max_scaled.train == [[13.0], [14.0]]
    assert transformed_data.quantile_transformed.train == [[17.0], [18.0]]
    assert transformed_data.max_abs_scaled.train == [[21.0], [22.0]]


# Patching the DataTransformer and SplitData
@patch('src.classify3D.transform_data.DataTransformer', autospec=True)
@patch('src.classify3D.split_data.SplitData', autospec=True)
def test_transform_data_initialization(
    mock_split_data: MagicMock, mock_data_transformer: MagicMock
) -> None:
    """
    Test the initialization of TransformData with mock SplitData object.

    Name
    ----
    test_transform_data_initialization

    Description
    -----------
    Test the initialization of TransformData with mock SplitData object.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

    Parameters
    ----------
    mock_split_data : MagicMock
        Mock SplitData object
    mock_data_transformer : MagicMock
        Mock DataTransformer object
    """
    # Create a mock SplitData object and its attributes
    mock_split_data_instance = mock_split_data.return_value
    mock_split_data_instance.X = MagicMock()
    mock_split_data_instance.y = MagicMock()

    # Mock DataTransformer instance and its return value from calling
    mock_data_transformer_instance = mock_data_transformer.return_value
    mock_transformed_data = MagicMock()  # Mock TransformedData
    mock_data_transformer_instance.return_value = mock_transformed_data

    # Initialize TransformData with the mock SplitData object
    transform_data = TransformData(split=mock_split_data_instance)

    # Assert that the DataTransformer instance was created
    mock_data_transformer.assert_called_once()

    # Assert that the DataTransformer's __call__ method was called
    mock_data_transformer_instance.assert_called_once_with(
        X=mock_split_data_instance.X
    )

    # Check that TransformData's attributes were correctly assigned
    assert transform_data.X == mock_transformed_data
    assert transform_data.y == mock_split_data_instance.y


def test_transform_data_repr() -> None:
    """
    Test the __repr__ method of TransformData.

    Name
    ----
    test_transform_data_repr

    Description
    -----------
    Test the __repr__ method of TransformData.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        transform_data
            Module provides classes and functions for transforming data.
    """
    # Mock objects for DataTransformer and TransformedData
    mock_transformer = MagicMock(spec=DataTransformer)
    mock_transformed_data = MagicMock()
    mock_y = MagicMock()

    # Initialize TransformData with mocked transformer and data
    transform_data = TransformData.__new__(TransformData)
    transform_data.transformer = mock_transformer
    transform_data.X = mock_transformed_data
    transform_data.y = mock_y

    # Call __repr__ and check the output
    result = repr(transform_data)
    assert isinstance(result, str)
    assert "TransformData" in result
    assert "transformer=" in result
    assert "X=" in result
    assert "y=" in result
