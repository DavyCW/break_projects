"""
Module provides classes and functions for transforming and scaling data.

Name
----
transform_data.py

Description
-----------
This module provides classes and functions for transforming and scaling data.

Dependencies
------------
    External
    --------
    sklearn
        A library for machine learning.

    Internal
    --------
    Data
        A class representing the split data.
    SplitData
        A class responsible for splitting the data.
    GenerateData
        Generates 3D data based on passed file.

Attributes
----------
    Classes
    -------
    TransformedData
        A class representing the transformed data.
    DataTransformer
        A class representing a transformer for data preprocessing.
    TransformData
        A class representing transformed data.
"""

from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    RobustScaler,
    MinMaxScaler,
    QuantileTransformer,
    MaxAbsScaler,
)
from src.classify3D.split_data import Data, SplitData
from src.classify3D.generate_data import GenerateData


class TransformedData:
    """
    A class representing the transformed data.

    Name
    ----
    TransformedData

    Description
    -----------
    This class represents the data transformed using different techniques. It
    provides attributes for storing the transformed data.

    Dependencies
    ------------
        Internal
        --------
        Data
            A class representing the split data.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the TransformedData with transformed data.
        __repr__()
            Provide a string representation of the TransformedData object.

        Variables
        ---------
        untransformed : Data
            The original untransformed data.
        standard_scaled : Data
            The data scaled using the standard scaler.
        power_transformed : Data
            The data scaled using the power transformer.
        robust_scaled : Data
            The data scaled using the robust scaler.
        min_max_scaled : Data
            The data scaled using the min-max scaler.
        quantile_transformed : Data
            The data scaled using the quantile transformer.
        max_abs_scaled : Data
            The data scaled using the max abs scaler.
    """

    def __init__(
        self,
        untransformed: Data,
        standard_scaled: Data,
        power_transformed: Data,
        robust_scaled: Data,
        min_max_scaled: Data,
        quantile_transformed: Data,
        max_abs_scaled: Data,
    ) -> None:
        """
        Initialize the TransformedData with transformed data.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the TransformedData object with the transformed data.

        Dependencies
        ------------
            Internal
            --------
            Data
                A class representing the split data.

        Parameters
        ----------
        untransformed : Data
            The original untransformed data.
        standard_scaled : Data
            The data scaled using the standard scaler.
        power_transformed : Data
            The data scaled using the power transformer.
        robust_scaled : Data
            The data scaled using the robust scaler.
        min_max_scaled : Data
            The data scaled using the min-max scaler.
        quantile_transformed : Data
            The data scaled using the quantile transformer.
        max_abs_scaled : Data
            The data scaled using the max abs scaler.

        Examples
        --------
        >>> transformed_data = TransformedData(
        ...                          untransformed=untransformed,
        ...                          standard_scaled=standard_scaled,
        ...                          power_transformed=power_transformed,
        ...                          robust_scaled=robust_scaled,
        ...                          min_max_scaled=min_max_scaled,
        ...                          quantile_transformed=quantile_transformed,
        ...                          max_abs_scaled=max_abs_scaled
        ...                          )
        """
        self.untransformed = untransformed
        self.standard_scaled = standard_scaled
        self.power_transformed = power_transformed
        self.robust_scaled = robust_scaled
        self.min_max_scaled = min_max_scaled
        self.quantile_transformed = quantile_transformed
        self.max_abs_scaled = max_abs_scaled

    def __repr__(self) -> str:
        """
        Provide a string representation of the TransformedData object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the TransformedData object.

        Returns
        -------
        str
            A string representation of the TransformedData object.

        Examples
        --------
        >>> print(transformed_data)
        TransformedData(untransformed=data,
        ...        standard_scaled=standard_scaled_data,
        ...        power_transformed=power_transformed_data,
        ...        robust_scaled=robust_scaled_data,
        ...        min_max_scaled=min_max_scaled_data,
        ...        quantile_transformed=quantile_transformed_data,
        ...        max_abs_scaled=max_abs_scaled_data
        ...        )
        """
        return (
            f"TransformedData(untransformed={self.untransformed}, "
            f"standard_scaled={self.standard_scaled}, "
            f"power_transformed={self.power_transformed}, "
            f"robust_scaled={self.robust_scaled}, "
            f"min_max_scaled={self.min_max_scaled}, "
            f"quantile_transformed={self.quantile_transformed}, "
            f"max_abs_scaled={self.max_abs_scaled})"
        )


class DataTransformer:
    """
    A class representing a transformer for data preprocessing.

    Name
    ----
    DataTransformer

    Description
    -----------
    This class represents a transformer for data preprocessing. It provides
    attributes and methods for scaling data using different techniques.


    Dependencies
    ------------
        External
        --------
        sklearn
            A library for machine learning.

        Internal
        --------
        Data
            A class representing the split data.
        TransformedData
            A class representing the transformed data.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the DataTransformer with scaling techniques.
        __repr__()
            Provide a string representation of the DataTransformer object.
        __call__()
            Transform the data using the specified techniques.

        Variables
        ---------
        standard_scaler : StandardScaler
            The standard scaler for transforming data.
        power_transformer : PowerTransformer
            The power transformer for transforming data
        robust_scaler : RobustScaler
            The robust scaler for transforming data.
        min_max_scaler : MinMaxScaler
            The min-max scaler for transforming data.
        quantile_transformer : QuantileTransformer
            The quantile transformer for transforming data.
        max_abs_scaler : MaxAbsScaler
            The max abs scaler for transforming data.
    """

    def __init__(self) -> None:
        """
        Initialize the DataTransformer with transforming techniques.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the DataTransformer object with the specified transforming
        techniques.

        Dependencies
        ------------
            External
            --------
            sklearn
                A library for machine learning.

        Examples
        --------
        >>> transformer = DataTransformer()
        """
        self.standard_scaler = StandardScaler()
        self.power_transformer = PowerTransformer()
        self.robust_scaler = RobustScaler()
        self.min_max_scaler = MinMaxScaler()
        self.quantile_transformer = QuantileTransformer()
        self.max_abs_scaler = MaxAbsScaler()

    def __repr__(self) -> str:
        """
        Provide a string representation of the DataTransformer object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the DataTransformer object.

        Returns
        -------
        str
            A string representation of the DataTransformer object.

        Examples
        --------
        >>> print(transformer)
        DataTransformer(standard_scaler=StandardScaler(),
        ...             power_transformer=PowerTransformer(),
        ...             robust_scaler=RobustScaler(),
        ...             min_max_scaler=MinMaxScaler(),
        ...             quantile_transformer=QuantileTransformer(),
        ...             max_abs_scaler=MaxAbsScaler()
        ...             )
        """
        return (
            f"DataTransformer(standard_scaler={self.standard_scaler}, "
            f"power_transformer={self.power_transformer}, "
            f"robust_scaler={self.robust_scaler}, "
            f"min_max_scaler={self.min_max_scaler}, "
            f"quantile_transformer={self.quantile_transformer}, "
            f"max_abs_scaler={self.max_abs_scaler})"
        )

    def __call__(self, X: Data) -> TransformedData:
        """
        Transform the data using the specified techniques.

        Name
        ----
        fit_transform

        Description
        -----------
        Transform the data using the specified techniques and return a
        TransformedData object.

        Dependencies
        ------------
            Internal
            --------
            Data
                A class representing the split data.
            TransformedData
                A class representing the transformed data.

        Parameters
        ----------
        X : Data
            The data to be transformed.

        Returns
        -------
        TransformedData
            A TransformedData object containing the transformed data.

        Examples
        --------
        >>> transformed_data = transformer(X=data)
        """
        # Fit the scalers to X.train and transform X.train, X.test, and X.val
        ss_tr = self.standard_scaler.fit_transform(X=X.train)
        pt_tr = self.power_transformer.fit_transform(X=X.train)
        rs_tr = self.robust_scaler.fit_transform(X=X.train)
        mm_tr = self.min_max_scaler.fit_transform(X=X.train)
        qt_tr = self.quantile_transformer.fit_transform(X=X.train)
        ma_tr = self.max_abs_scaler.fit_transform(X=X.train)

        # Transform X.test and X.val using the already fitted scalers
        ss_t = self.standard_scaler.transform(X=X.test)
        pt_t = self.power_transformer.transform(X=X.test)
        rs_t = self.robust_scaler.transform(X=X.test)
        mm_t = self.min_max_scaler.transform(X=X.test)
        qt_t = self.quantile_transformer.transform(X=X.test)
        ma_t = self.max_abs_scaler.transform(X=X.test)

        ss_v = self.standard_scaler.transform(X=X.val)
        pt_v = self.power_transformer.transform(X=X.val)
        rs_v = self.robust_scaler.transform(X=X.val)
        mm_v = self.min_max_scaler.transform(X=X.val)
        qt_v = self.quantile_transformer.transform(X=X.val)
        ma_v = self.max_abs_scaler.transform(X=X.val)

        # Create new Data objects for each transformation
        standard_scaled_data = Data(train=ss_tr, test=ss_t, val=ss_v)
        power_transformed_data = Data(train=pt_tr, test=pt_t, val=pt_v)
        robust_scaled_data = Data(train=rs_tr, test=rs_t, val=rs_v)
        min_max_scaled_data = Data(train=mm_tr, test=mm_t, val=mm_v)
        quantile_transformed_data = Data(train=qt_tr, test=qt_t, val=qt_v)
        max_abs_scaled_data = Data(train=ma_tr, test=ma_t, val=ma_v)

        # Create ScaledData object with all the scaled datasets
        scaled_data = TransformedData(
            untransformed=X,
            standard_scaled=standard_scaled_data,
            power_transformed=power_transformed_data,
            robust_scaled=robust_scaled_data,
            min_max_scaled=min_max_scaled_data,
            quantile_transformed=quantile_transformed_data,
            max_abs_scaled=max_abs_scaled_data,
        )

        return scaled_data


class TransformData:
    """
    A class representing transformed data.

    Name
    ----
    TransformData

    Description
    -----------
    This class represents transformed data. It provides attributes for storing
    transformed data.

    Dependencies
    ------------
        Internal
        --------
        Data
            A class representing the split data.
        SplitData
            A class responsible for splitting the data.
        TransformedData
            A class representing the transformed data.
        DataTransformer
            A class representing a transformer for data preprocessing.

    Attributes
    ----------
        Functions
        ---------
        __init__()
            Initialize the TransformData with a SplitData object.
        __repr__()
            Provide a string representation of the TransformData object.

        Variables
        ---------
        transformer : DataTransformer
            The transformer used to transform the data.
        X : TransformedData
            The transformed data.
        y : Data
            The target data.
    """

    def __init__(self, split: SplitData) -> None:
        """
        Initialize the TransformData with a SplitData object.

        Name
        ----
        __init__

        Description
        -----------
        Initialize the TransformData object with a SplitData object.

        Dependencies
        ------------
            SplitData
                A class responsible for splitting the data.
            DataTransformer
                A class representing a transformer for data preprocessing.

        Parameters
        ----------
        split : SplitData
            The SplitData object containing the data to be transformed.


        Examples
        --------
        >>> transform_data = TransformData(split=split_data)
        """
        # Initialize the scalers
        self.transformer = DataTransformer()
        self.X = self.transformer(X=split.X)
        self.y = split.y

    def __repr__(self) -> str:
        """
        Provide a string representation of the TransformData object.

        Name
        ----
        __repr__

        Description
        -----------
        Provide a string representation of the TransformData object.

        Returns
        -------
        str
            A string representation of the TransformData object.

        Examples
        --------
        >>> print(transform_data)
        TransformData(transformer=DataTransformer(),
        ...           X=TransformedData(),
        ...           y=Data()
        ...           )
        """
        return (
            f"TransformData(transformer={self.transformer}, "
            f"X={self.X}, "
            f"y={self.y})"
        )


if __name__ == "__main__":
    gen = GenerateData()
    gen('container')
    split = SplitData(gen=gen)
    transform = TransformData(split=split)
