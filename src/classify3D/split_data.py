"""
Module contains functions and classes for splitting data.

Name
----
split_data.py

Description
-----------
This module provides functionality for splitting data into training,
validation, and testing sets using the `train_test_split` function from
scikit-learn.

Dependencies
------------
    External
    --------
    numpy
        A library for efficient numerical computation.
    sklearn
        A library for machine learning.

    Internal
    --------
    GenerateData
        Generates 3D data based on passed file.

Attributes
----------
    Classes
    -------
    Data
        A class representing the data to be split.
    SplitData
        A class responsible for splitting the data.
"""

from sklearn.model_selection import train_test_split
import numpy as np
from src.classify3D.classify3d import GenerateData  # type: ignore


class Data:
    """
    A class representing the data split.

    Name
    ----
    Data

    Description
    -----------
    This class represents the data split into training, validation, and
    testing sets. It provides attributes for storing the training, testing,
    and validation data.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the Data with training, testing, and validation data.
        __repr__()
            Provide a string representation of the Data object.

        Variables
        ---------
        train : np.ndarray
            The training data.
        test : np.ndarray
            The testing data.
        val : np.ndarray
            The validation data.
    """

    def __init__(
        self, train: np.ndarray, test: np.ndarray, val: np.ndarray
    ) -> None:
        """
        Initialize the Data with training, testing, and validation data.

        Name
        ----
        __init__

        Description
        -----------
        Initialize a Data object with training, testing, and validation data.

        Dependencies
        ------------
            External
            --------
            numpy
                A library for efficient numerical computation.

        Parameters
        ----------
        train : np.ndarray
            The training data.
        test : np.ndarray
            The testing data.
        val : np.ndarray
            The validation data.

        Examples
        --------
        >>> import numpy as np
        >>> train_data = np.array([1, 2, 3])
        >>> test_data = np.array([4, 5, 6])
        >>> val_data = np.array([7, 8, 9])
        >>> data = Data(train_data, test_data, val_data)
        """
        self.train = train
        self.test = test
        self.val = val

    def __repr__(self) -> str:
        """
        Provide a string representation of the Data object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the Data object with the parameter
        shapes.

        Returns
        -------
        str
            A string representation of the Data object.

        Examples
        --------
        >>> import numpy as np
        >>> train_data = np.array([1, 2, 3])
        >>> test_data = np.array([4, 5, 6])
        >>> val_data = np.array([7, 8, 9])
        >>> data = Data(train_data, test_data, val_data)
        >>> print(data)
        Data(train=(3, )),
             test=(3, )),
             val=(3, ))
             )
        """
        return (
            f"Data(train: {self.train.shape}, "
            f"test: {self.test.shape}, "
            f"val: {self.val.shape})"
        )


class SplitData:
    """
    A class responsible for splitting the data.

    Name
    ----
    SplitData

    Description
    -----------
    This class provides methods for splitting data into training, testing, and
    validation sets. It uses the Data class to store the split data and
    provides attributes for accessing the split data.

    Dependencies
    ------------
        External
        --------
        numpy
            A library for efficient numerical computation.
        sklearn
            A library for machine learning.

        Internal
        --------
        Data
            A class representing the data split.
        GenerateData
            Generates 3D data based on passed file.

    Attributes
    ----------
        Classes
        -------
        X
            The feature data.
        y
            The target data.

        Functions
        ---------
        __init__()
            Initialize the SplitData object with the data to be split.
        __repr__()
            Provide a string representation of the SplitData object.
        _check_props()
            Check if the proportions for training and testing are valid.
        _split_data()
            Split the data into training, testing, and validation sets.

        Variables
        ---------
        train_prop : float
            The proportion of data to use for training.
        test_prop : float
            The proportion of data to use for testing.
        val_prop : float
            The proportion of data to use for validation.
    """

    def __init__(
        self,
        gen: GenerateData,
        train_prop: float = 0.7,
        test_prop: float = 0.15,
    ) -> None:
        """
        Initialize the SplitData object with the data to be split.

        Name
        ----
        __init__

        Description
        -----------
        This method initializes the SplitData object by setting the feature
        data (X), target data (y), and the proportions for training and
        testing. It also calculates the proportion for validation and checks if
        the proportions are valid.

        Dependencies
        ------------
            External
            --------
            numpy
                A library for efficient numerical computation.

            Internal
            --------
            GenerateData
                Generates 3D data based on passed file.

        Parameters
        ----------
        gen : np.GenerateData
            The data.
        train_prop : float, optional
            The proportion of data to use for training (default is 0.7).
        test_prop : float, optional
            The proportion of data to use for testing (default is 0.15).

        Examples
        --------
        >>> gen = GenerateData()
        >>> split_data = SplitData(gen, train_prop=0.6, test_prop=0.2)
        """
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = max(1 - self.train_prop - self.test_prop, 0)
        self._check_props()
        self._split_data(gen=gen)

    def __repr__(self) -> str:
        """
        Provide a string representation of the SplitData object.

        Name
        ----
        __repr__

        Description
        -----------
        This method returns a string representation of the SplitData object,
        including the proportions for training, testing, and validation.

        Returns
        -------
        str
            A string representation of the SplitData object.

        Examples
        --------
        >>> gen = GenerateData()
        >>> split_data = SplitData(gen)
        >>> print(split_data)
        SplitData(train_prop: 0.7, val_prop: 0.15, test_prop: 0.15)
        """
        return (
            f"SplitData(train_prop: {self.train_prop}, "
            f"val_prop: {self.val_prop}, "
            f"test_prop: {self.test_prop})"
        )

    def _check_props(self) -> None:
        """
        Check if the proportions for training and testing are valid.

        Name
        ----
        _check_props

        Description
        -----------
        This method checks if the proportions for training and testing are
        valid (i.e., between 0 and 1) and if their sum is not greater than 1.

        Raises
        ------
        ValueError
            If the proportions are not valid.

        Examples
        --------
        >>> gen = GenerateData()
        >>> split_data = SplitData(gen)
        >>> split_data.train_prop = 0.8
        >>> split_data.test_prop = 0.3
        >>> split_data._check_props()
        ValueError: Invalid proportions: train_prop + test_prop > 1
        """
        if not (0 <= self.train_prop <= 1 and 0 <= self.test_prop <= 1):
            raise ValueError(
                "Invalid proportions: " "train_prop or test_prop out of range"
            )
        if self.train_prop + self.test_prop > 1:
            raise ValueError("Invalid proportions: train_prop + test_prop > 1")

    def _split_data(self, gen: GenerateData) -> None:
        """
        Split the data into training, testing, and validation sets.

        Name
        ----
        _split_data

        Description
        -----------
        The data is split based on the proportions specified in the
        `train_prop`, `val_prop`, and `test_prop` attributes.

        Dependencies
        ------------
            External
            --------
            numpy
                A library for efficient numerical computation.
            sklearn
                A library for machine learning.

            Internal
            --------
            Data
                A class representing the data split.
            GenerateData
                Generates 3D data based on passed file.

        Parameters
        ----------
        gen : np.GenerateData
            The data.

        Examples
        --------
        >>> gen = GenerateData()
        >>> split_data = SplitData(gen)
        >>> split_data._split_data(gen)
        """
        # First split: Divide into training and a combined test/validation set
        X_tr, X_tv, y_tr, y_tv = train_test_split(
            gen.X, gen.y, test_size=(1 - self.train_prop)
        )
        # Second split: Divide the combined test/validation set
        if self.val_prop > 0:
            X_t, X_v, y_t, y_v = train_test_split(
                X_tv,
                y_tv,
                test_size=(self.test_prop / (self.test_prop + self.val_prop)),
            )
            self.X = Data(train=X_tr, test=X_t, val=X_v)
            self.y = Data(train=y_tr, test=y_t, val=y_v)
        else:
            self.X = Data(train=X_tr, test=X_tv, val=np.empty_like(X_tr))
            self.y = Data(train=y_tr, test=y_tv, val=np.empty_like(y_tr))
