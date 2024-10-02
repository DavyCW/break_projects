"""
Test suite for the Outliers class.

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
    outliers
        Module for outliers detection.

Attributes
----------
    Functions
    ---------
    x_train()
        Fixture for reusable data.
    outliers()
        Fixture for Outliers object.
    test_z_score_outliers()
        Test Z-Score outliers detection.
    test_iqr_outliers()
        Test IQR outliers detection.
    test_isolation_forest_outliers()
        Test IsolationForest outliers detection.
    test_one_class_svm_outliers()
        Test OneClassSVM outliers detection.
    test_local_outlier_factor_outliers()
        Test LocalOutlierFactor outliers detection.
    test_get_outliers()
        Test _get_outliers method output.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.classify3D.outliers import Outliers  # type: ignore
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
def outliers(X_train: np.ndarray) -> Outliers:
    """
    Fixture for Outliers object.

    Name
    ----
    outliers

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

        Internal
        --------
        outliers
            Module for outliers detection.

    Parameters
    ----------
    X_train : np.ndarray
        Array of training data.

    Returns
    -------
    Outliers
        Object for outliers detection.
    """
    return Outliers(X_train)


def test_z_score_outliers(outliers: Outliers) -> None:
    """
    Test Z-Score outliers detection.

    Name
    ----
    test_z_score_outliers

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
        outliers
            Module for outliers detection.

    Parameters
    ----------
    outliers : Outliers
        Object for outliers detection.
    """
    result = outliers._get_z_score_outliers()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 100  # Should match X_train


def test_iqr_outliers(outliers: Outliers) -> None:
    """
    Test IQR outliers detection.

    Name
    ----
    test_iqr_outliers

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
    outliers : Outliers
        Object for outliers detection.
    """
    result = outliers._get_iqr_outliers()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 100  # Should match X_train


@patch('src.classify3D.outliers.IsolationForest')
def test_isolation_forest_outliers(
    mock_isolation_forest: MagicMock, outliers: Outliers
) -> None:
    """
    Test IsolationForest outliers detection with mock.

    Name
    ----
    test_isolation_forest_outliers

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
        outliers
            Module for outliers detection.

    Parameters
    ----------
    mock_isolation_forest : MagicMock
        Mock object for LocalOutlierFactor.
    outliers : Outliers
        Object for outliers detection.
    """
    mock_model = mock_isolation_forest.return_value
    mock_model.decision_function.return_value = np.random.rand(100) - 0.5
    result = outliers._get_isolation_forest_outliers()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 100  # Should match X_train
    mock_isolation_forest.assert_called_once()


@patch('src.classify3D.outliers.OneClassSVM')
def test_one_class_svm_outliers(
    mock_one_class_svm: MagicMock, outliers: Outliers
) -> None:
    """
    Test OneClassSVM outliers detection with mock.

    Name
    ----
    test_one_class_svm_outliers

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
        outliers
            Module for outliers detection.

    Parameters
    ----------
    mock_one_class_svm : MagicMock
        Mock object for LocalOutlierFactor.
    outliers : Outliers
        Object for outliers detection.
    """
    mock_model = mock_one_class_svm.return_value
    mock_model.decision_function.return_value = np.random.rand(100) - 0.5
    result = outliers._get_one_class_svm_outliers()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 100  # Should match X_train
    mock_one_class_svm.assert_called_once()


@patch('src.classify3D.outliers.LocalOutlierFactor')
def test_local_outlier_factor_outliers(
    lof: MagicMock, outliers: Outliers
) -> None:
    """
    Test LocalOutlierFactor outliers detection with mock.

    Name
    ----
    test_local_outlier_factor_outliers

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
        outliers
            Module for outliers detection.

    Parameters
    ----------
    lof : MagicMock
        Mock object for LocalOutlierFactor.
    outliers : Outliers
        Object for outliers detection.
    """
    mock_model = lof.return_value
    mock_model.decision_function.return_value = np.random.rand(100) - 0.5
    result = outliers._get_local_outlier_factor_outliers()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 100  # Should match X_train
    lof.assert_called_once()


def test_get_outliers(outliers: Outliers) -> None:
    """
    Test _get_outliers method output.

    Name
    ----
    test_get_outliers

    Dependencies
    ------------
        External
        --------
        numpy
            Package for fast array manipulation.

        Internal
        --------
        outliers
            Module for outliers detection.

    Parameters
    ----------
    outliers : Outliers
        Object for outliers detection.
    """
    outliers_result = np.array(object=[True, False, True, False] * 25)
    fig = outliers._get_outliers('test_method', outliers_result)
    assert fig is not None
    assert isinstance(fig, go.Figure)  # Check if it returns a Plotly figure
