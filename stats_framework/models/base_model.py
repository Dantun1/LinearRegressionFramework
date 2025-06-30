from abc import ABC, abstractmethod
from typing import Optional
from stats_framework.data import BaseDataset

class BaseModel(ABC):

    def __init__(self, dataset: BaseDataset):
        self._dataset: Optional[BaseDataset] = None
        self.dataset= dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: BaseDataset):
        if not isinstance(dataset, BaseDataset):
            raise TypeError("The provided object is not a valid dataset. The provided dataset must be a subclass of BaseDataset")
        self._dataset = dataset

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, *args):
        pass
