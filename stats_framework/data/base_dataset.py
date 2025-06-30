from abc import ABC, abstractmethod
import pandas as pd

class BaseDataset(ABC):
    """
    This is an abstract base class for dataset objects within the framework.

    This class defines the basic interface that all dataset classes must implement, ensuring
    all other components like models or transformations can operate on any dataset object consistently.
    All subclasses should implement their own custom validation depending on their specific use case and required
    format.
    """

    @property
    @abstractmethod
    def features(self) -> pd.DataFrame:
        """
        Return features as a pandas DataFrame.
        """
        pass

    @property
    @abstractmethod
    def response(self) -> pd.Series:
        """
        Return response variable as a pandas Series.
        """
        pass

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """
        Return a list of feature names.
        """
        pass

    @property
    @abstractmethod
    def response_name(self) -> str:
        """
        Return name of response variable.
        """
        pass


