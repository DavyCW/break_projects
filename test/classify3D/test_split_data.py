"""
Module contains functions and classes for testing the split_data module.

Name
----
test_split_data.py

Description
-----------
This module contains functions and classes for testing the split_data module.

Dependencies
------------
    External
    --------
    numpy
        A library for efficient numerical computation.
    pytest
        A library for writing and running tests.

    Internal
    --------
    generate_data
        Module containing the GenerateData class.
    split_data
        Module contains functions and classes for splitting data.

Attributes
----------
    Functions
    ---------
    test_data_initialization()
        Test that Data class correctly initializes with given numpy arrays.
    test_data_representation()
        Test that Data class __repr__ returns the expected representation.
    test_split_data_initialization_valid_proportions()
        Test that SplitData class initializes correctly with valid proportions.
    test_split_data_invalid_proportions()
        Test that SplitData raises a ValueError for invalid proportions.
    test_split_data_repr()
        Test that SplitData __repr__ returns the correct string format.
    test_split_data_split_correctly()
        Test that data is split correctly.
    test_split_data_no_validation()
        Test that SplitData handles no validation data.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.classify3D.split_data import Data, SplitData
from src.classify3D.generate_data import GenerateData


def test_data_initialization() -> None:
    """
    Test that Data class correctly initializes with given numpy arrays.

    Name
    ----
    test_data_initialization

    Description
    -----------
    Test that Data class correctly initializes with given numpy arrays.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.

        Internal
        --------
        split_data
            Module contains functions and classes for splitting data.
    """
    train = np.array(object=[1, 2, 3])
    test = np.array(object=[4, 5, 6])
    val = np.array(object=[7, 8, 9])

    data = Data(train=train, test=test, val=val)

    # Ensure the attributes are correctly set
    assert np.array_equal(a1=data.train, a2=train)
    assert np.array_equal(a1=data.test, a2=test)
    assert np.array_equal(a1=data.val, a2=val)


def test_data_representation() -> None:
    """
    Test that Data class __repr__ returns the expected representation.

    Name
    ----
    test_data_representation

    Description
    -----------
    Test that Data class __repr__ returns the expected representation.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.

        Internal
        --------
        split_data
            Module contains functions and classes for splitting data.
    """
    train = np.array(object=[1, 2, 3])
    test = np.array(object=[4, 5, 6])
    val = np.array(object=[7, 8, 9])

    data = Data(train=train, test=test, val=val)

    expected_repr = "Data(train: (3,), test: (3,), val: (3,))"

    # Check if the string representation is correct
    assert repr(data) == expected_repr


def test_split_data_initialization_valid_proportions() -> None:
    """
    Test that SplitData class initializes correctly.

    Name
    ----
    test_split_data_initialization_valid_proportions

    Description
    -----------
    Test that SplitData class initializes correctly with valid proportions.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pytest
            A library for writing and running tests.

        Internal
        --------
        generate_data
            Module containing the GenerateData class.
        split_data
            Module contains functions and classes for splitting data.
    """
    gen_mock = MagicMock(spec=GenerateData)
    gen_mock.X = np.random.rand(100, 3)
    gen_mock.y = np.random.rand(100)

    split = SplitData(gen=gen_mock, train_prop=0.6, test_prop=0.2)

    assert split.train_prop == 0.6
    assert split.test_prop == 0.2
    assert split.val_prop == 0.2


def test_split_data_invalid_proportions() -> None:
    """
    Test that SplitData raises a ValueError for invalid proportions.

    Name
    ----
    test_split_data_invalid_proportions

    Description
    -----------
    Test that SplitData raises a ValueError for invalid proportions.

    Dependencies
    ------------
        External
        --------
        pytest
            A library for writing and running tests.

        Internal
        --------
        generate_data
            Module containing the GenerateData class.
        split_data
            Module contains functions and classes for splitting data.
    """
    gen_mock = MagicMock(spec=GenerateData)

    # Proportions exceed 1
    with pytest.raises(
        expected_exception=ValueError, match="Invalid proportions"
    ):
        SplitData(gen=gen_mock, train_prop=0.8, test_prop=0.3)


def test_split_data_repr() -> None:
    """
    Test that SplitData __repr__ returns the correct string format.

    Name
    ----
    test_split_data_repr

    Description
    -----------
    Test that SplitData __repr__ returns the correct string format.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pytest
            A library for writing and running tests.

        Internal
        --------
        generate_data
            Module containing the GenerateData class.
        split_data
            Module contains functions and classes for splitting data.
    """
    gen_mock = MagicMock(spec=GenerateData)
    gen_mock.X = np.random.rand(100, 3)
    gen_mock.y = np.random.rand(100)
    split = SplitData(gen=gen_mock, train_prop=0.8, test_prop=0.2)

    expected_repr = "SplitData(train_prop: 0.8, val_prop: 0, test_prop: 0.2)"

    # Check if the string representation is correct
    assert repr(split) == expected_repr


def test_split_data_split_correctly() -> None:
    """
    Test that data is split correctly.

    Name
    ----
    test_split_data_split_correctly

    Description
    -----------
    Test that data is split correctly.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pytest
            A library for writing and running tests.

        Internal
        --------
        generate_data
            Module containing the GenerateData class.
        split_data
            Module contains functions and classes for splitting data.
    """
    gen_mock = MagicMock(spec=GenerateData)
    gen_mock.X = np.random.rand(100, 3)
    gen_mock.y = np.random.rand(100)

    split = SplitData(gen=gen_mock, train_prop=0.6, test_prop=0.2)

    # Check if the splits are of the correct size
    assert split.X.train.shape[0] == 60
    assert split.X.test.shape[0] == 20
    assert split.X.val.shape[0] == 20

    assert split.y.train.shape[0] == 60
    assert split.y.test.shape[0] == 20
    assert split.y.val.shape[0] == 20


def test_split_data_no_validation() -> None:
    """
    Test that SplitData handles no validation data.

    Name
    ----
    test_split_data_no_validation

    Description
    -----------
    Test that SplitData handles no validation data.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        pytest
            A library for writing and running tests.

        Internal
        --------
        generate_data
            Module containing the GenerateData class.
        split_data
            Module contains functions and classes for splitting data.
    """
    gen_mock = MagicMock(spec=GenerateData)
    gen_mock.X = np.random.rand(100, 3)
    gen_mock.y = np.random.rand(100)

    split = SplitData(gen=gen_mock, train_prop=0.8, test_prop=0.2)

    # Check if the validation set is empty
    assert split.X.val.size == 0
    assert split.y.val.size == 0
