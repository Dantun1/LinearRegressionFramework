from dataclasses import dataclass, InitVar, field
import numpy as np
import pandas as pd
from functools import singledispatchmethod
from typing import Union
from base_dataset import BaseDataset

@dataclass
class LinearRegressionDataset(BaseDataset):
    """
    A dataclass for linear regression datasets that processes and stores the features and response data.

    Args:
        features_input: The unvalidated features input
        response_input: The unvalidated response input

    Attributes:
        _features: A validated pandas DataFrame of features.
        _response: A validated pandas Series of responses.
    """
    features_input: InitVar[Union[np.ndarray, pd.DataFrame, pd.Series]]
    response_input: InitVar[Union[np.ndarray, pd.DataFrame, pd.Series]]
    _features:  pd.DataFrame = field(init=False, repr=False)
    _response: pd.Series = field(init=False, repr=False)



    @property
    def features(self):
        return self._features

    @property
    def feature_names(self):
        return self._features.columns.to_list()

    @property
    def response(self):
        return self._response

    @property
    def response_name(self):
        return self._response.name


    def __post_init__(self, features_input, response_input):
        """
        Validates the dataset by ensuring
        1) features and response are numpy arrays, pandas dataframes or series with real number types.
        2) features and response have the same length
        """
        self._features = self._process_features(features_input)
        self._response = self._process_response(response_input)

        if len(self.features) != len(self.response):
            raise ValueError("Features and response must have the same length")

    @singledispatchmethod
    def _process_features(self, features):
        # Reject invalid feature input types
        raise TypeError(f"Unsupported features type {type(features)}.\nFeatures must be a numpy array, pandas dataframe or pandas series.")

    @_process_features.register(pd.DataFrame)
    def _(self, features_df):
        self._validate_real_types(features_df)
        return features_df

    @_process_features.register(pd.Series)
    def _(self, features_s):
        self._validate_real_types(features_s)
        # Standardize pd.Series to dataframe
        return pd.Series.to_frame(features_s)

    @_process_features.register(np.ndarray)
    def _(self, features_np):
        # Reject invalid np arrays and standardize 1D numpy arrays to (n,1) matrix
        if features_np.ndim > 2:
            raise ValueError("Features as np array must have 1 or 2 dimensions.")
        elif features_np.ndim == 1:
            features_np = features_np.reshape(-1, 1)

        col_names = [f"x{i}" for i in range(features_np.shape[1])]
        features_df = pd.DataFrame(features_np, columns=col_names)

        self._validate_real_types(features_df)
        return features_df

    @singledispatchmethod
    def _process_response(self, response):
        raise TypeError(f"Unsupported response type {type(response)}. Response must be a 1D numpy array, single-column pandas dataframe or pandas series")

    @_process_response.register(pd.DataFrame)
    def _(self, response_df):
        if response_df.shape[1] > 1:
            raise ValueError("Response dataframe must have 1 column.")
        self._validate_real_types(response_df)
        return response_df.squeeze()

    @_process_response.register(np.ndarray)
    def _(self, response_np):

        # Reject all numpy arrays with more than 2 dimensions or 2-dimensional vectors with more than 1 column. Squeeze 2D with single column.
        if response_np.ndim > 2 or (response_np.ndim == 2 and response_np.shape[1] != 1):
            raise ValueError("Response array must be 1D numpy array or single-column 2D np array")
        elif response_np.ndim == 2:
            response_np = np.squeeze(response_np)

        response_s = pd.Series(response_np, name="y")
        self._validate_real_types(response_np)
        return response_s

    @_process_response.register(pd.Series)
    def _(self, response_s):

        self._validate_real_types(response_s)
        return response_s

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

    #TODO: Handle Nans, if provided with Nans, remove by default unless specified in constructor.
    #TODO: Add names variable to store column names if provided


if __name__ == "__main__":
    ...