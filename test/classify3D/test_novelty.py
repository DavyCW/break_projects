"""
Test suite for the Novelty class.

Name
----
test_novelty.py

Dependencies
------------
    External
    --------
    pytest
        Package for testing.
    numpy
        Package for fast array manipulation.
    plotly
        Package for plotting.

    Internal
    --------
    novelty
        Module for novelty detection.

Attributes
----------
    Functions
    ---------
    x_train()
        Fixture for reusable data.
    x_test()
        Fixture for reusable data.
    novelty()
        Fixture for Novelty object.
    test_z_score_novelty()
        Test Z-Score novelty detection.
    test_iqr_novelty()
        Test IQR novelty detection.
    test_isolation_forest_novelty()
        Test IsolationForest novelty detection.
    test_one_class_svm_novelty()
        Test OneClassSVM novelty detection.
    test_local_outlier_factor_novelty()
        Test LocalOutlierFactor novelty detection.
    test_get_novelty()
        Test _get_novelty method output.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.classify3D.novelty import Novelty  # type: ignore
from plotly import graph_objects as go


@pytest.fixture
def X_train() -> np.ndarray:
    """
    Fixture for reusable data.

    Name
    ----
    X_train

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

    Returns
    -------
    np.ndarray
        Array of train data.
    """
    return np.random.rand(100, 6)


@pytest.fixture
def X_test() -> np.ndarray:
    """
    Fixture for reusable data.

    Name
    ----
    X_test

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

    Returns
    -------
    np.ndarray
        Array of test data.
    """
    return np.random.rand(20, 6)


@pytest.fixture
def novelty(X_train: np.ndarray, X_test: np.ndarray) -> Novelty:
    """
    Fixture for Novelty object.

    Name
    ----
    novelty

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    X_train : np.ndarray
        Array of training data.
    X_test : np.ndarray
        Array of test data.

    Returns
    -------
    Novelty
        Object for novelty detection.
    """
    return Novelty(X_train, X_test)


def test_z_score_novelty(novelty: Novelty) -> None:
    """
    Test Z-Score novelty detection.

    Name
    ----
    test_z_score_novelty

    Dependencies
    ------------
        External
        --------
        pytest
            Package for testing.
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    novelty : Novelty
        Object for novelty detection.
    """
    result = novelty._get_z_score_novelty()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 20  # Should match X_test


def test_iqr_novelty(novelty: Novelty) -> None:
    """
    Test IQR novelty detection.

    Name
    ----
    test_iqr_novelty

    Dependencies
    ------------
        External
        --------
        pytest
            Package for testing.
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    novelty : Novelty
        Object for novelty detection.
    """
    result = novelty._get_iqr_novelty()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 20  # Should match X_test


@patch('src.classify3D.novelty.IsolationForest')
def test_isolation_forest_novelty(
    mock_isolation_forest: MagicMock, novelty: Novelty
) -> None:
    """
    Test IsolationForest novelty detection with mock.

    Name
    ----
    test_isolation_forest_novelty

    Dependencies
    ------------
        External
        --------
        pytest
            Package for testing.
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    mock_isolation_forest : MagicMock
        Mock object for LocalOutlierFactor.
    novelty : Novelty
        Object for novelty detection.
    """
    mock_model = mock_isolation_forest.return_value
    mock_model.decision_function.return_value = np.random.rand(20) - 0.5
    result = novelty._get_isolation_forest_novelty()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 20  # Should match X_test
    mock_isolation_forest.assert_called_once()


@patch('src.classify3D.novelty.OneClassSVM')
def test_one_class_svm_novelty(
    mock_one_class_svm: MagicMock, novelty: Novelty
) -> None:
    """
    Test OneClassSVM novelty detection with mock.

    Name
    ----
    test_one_class_svm_novelty

    Dependencies
    ------------
        External
        --------
        pytest
            Package for testing.
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    mock_one_class_svm : MagicMock
        Mock object for LocalOutlierFactor.
    novelty : Novelty
        Object for novelty detection.
    """
    mock_model = mock_one_class_svm.return_value
    mock_model.decision_function.return_value = np.random.rand(20) - 0.5
    result = novelty._get_one_class_svm_novelty()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 20  # Should match X_test
    mock_one_class_svm.assert_called_once()


@patch('src.classify3D.novelty.LocalOutlierFactor')
def test_local_outlier_factor_novelty(
    lof: MagicMock, novelty: Novelty
) -> None:
    """
    Test LocalOutlierFactor novelty detection with mock.

    Name
    ----
    test_local_outlier_factor_novelty

    Dependencies
    ------------
        External
        --------
        pytest
            Package for testing.
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    lof : MagicMock
        Mock object for LocalOutlierFactor.
    novelty : Novelty
        Object for novelty detection.
    """
    mock_model = lof.return_value
    mock_model.decision_function.return_value = np.random.rand(20) - 0.5
    result = novelty._get_local_outlier_factor_novelty()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 20  # Should match X_test
    lof.assert_called_once()


def test_get_novelty(novelty: Novelty) -> None:
    """
    Test _get_novelty method output.

    Name
    ----
    test_get_novelty

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

        Internal
        --------
        novelty
            Module for novelty detection.

    Parameters
    ----------
    novelty : Novelty
        Object for novelty detection.
    """
    novelty_result = np.array(object=[True, False, True, False] * 5)
    fig = novelty._get_novelty('test_method', novelty_result)
    assert fig is not None
    assert isinstance(fig, go.Figure)  # Check if it returns a Plotly figure
