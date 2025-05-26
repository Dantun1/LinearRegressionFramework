from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Union

@dataclass
class LinearRegressionDataset:
    """
    A dataclass for linear regression datasets that holds the features and response

    Args:
        _features: A numpy array, pandas DataFrame or pandas Series of features - standardized to a 2D array or DataFrame post init.
        _response: A numpy array, pandas DataFrame or pandas Series of response - standardized to a 1D array or Series post init.
    """
    _features: Union[np.ndarray,pd.DataFrame,pd.Series]
    _response: Union[np.ndarray,pd.DataFrame, pd.Series]

    @property
    def features(self):
        return self._features

    @property
    def response(self):
        return self._response

    def __post_init__(self):
        """
        Validates the dataset by ensuring
        1) features and response are numpy arrays, pandas dataframes or series with real number types.
        2) features and response have the same length
        """
        self._validate_and_standardize_features()
        self._validate_and_standardize_response()

        if len(self._features) != len(self._response):
            raise ValueError("Features and response must have the same length")


    def _validate_and_standardize_features(self ):
        # Reject invalid input types
        if not isinstance(self._features, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("Features must be a numpy array, pandas dataframe or pandas series")

        # Reject all numpy arrays with more than 2 dimensions
        if isinstance(self._features, np.ndarray) and self._features.ndim > 2:
            raise ValueError("Features must be a 1D or 2D numpy array, pandas dataframe or pandas series")

        # Standardize pandas series to dataframe
        if isinstance(self._features, pd.Series):
            self._features = self._features.to_frame()

        # Standardize (n,) numpy arrays to (n,1) matrix
        if isinstance(self._features, np.ndarray) and self._features.ndim == 1:
            self._features = self._features.reshape(-1, 1) # Standardize vectors to single column matrix


        self._validate_real_types(self._features)

    def _validate_and_standardize_response(self):
        valid_response_message = "Response must be a 1D numpy array, single-column pandas dataframe or pandas series"

        # Reject invalid input types
        if not isinstance(self._response, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("Response must be a numpy array, pandas dataframe or pandas series")

        # Reject all numpy arrays with more than 2 dimensions or 2-dimensional vectors with more than 1 column
        if isinstance(self._response, np.ndarray) and (self._response.ndim >2 or (self._response.ndim == 2 and self._response.shape[1] != 1)) :
            raise ValueError(valid_response_message)
        # Reject pandas dataframes with more than 1 column
        if isinstance(self._response, pd.DataFrame) and self._features.shape[1] > 1:
            raise ValueError(valid_response_message)

        # Standardize pandas dataframes to series
        if isinstance(self._response, pd.DataFrame):
            self._response = self._response.squeeze()

        # Standardize (n,1) numpy arrays to 1d vector
        if isinstance(self._response, np.ndarray) and self._response.ndim == 2:
            self._response = self._response.squeeze()


        self._validate_real_types(self._response)

    @staticmethod
    def _validate_real_types(data):

        # Validate numpy array or pandas series values
        if hasattr(data, "dtype"):
            if not np.issubdtype(data.dtype, np.integer) and not np.issubdtype(data.dtype, np.floating):
                raise TypeError("Data must consist of real numbers")

        # Validate pandas dataframe values
        if isinstance(data, pd.DataFrame):
            for dtype in data.dtypes:
                if not np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.floating):
                    raise TypeError("Data must consist of real numbers")
    def __str__(self):
        return f"LinearRegressionDataset with dimensions-> Features: {self._features.shape}, Response: {self._response.shape}"

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(features={self._features.shape}, response={self._response.shape})"

    #TODO: Handle Nans




class DataValidation():
    ...

if __name__ == "__main__":
    x= np.random.rand(5) * 10
    y = pd.Series(np.random.rand(5) * 10)
    dataset = LinearRegressionDataset(x, y)
    print(dataset.response)
