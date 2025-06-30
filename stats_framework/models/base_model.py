from abc import ABC, abstractmethod
from stats_framework.data import BaseDataset

class Model(ABC):

    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if not isinstance(dataset, BaseDataset):
            raise TypeError("The provided object is not a valid dataset. Dataset must be a subclass of BaseDataset")
        self._dataset = dataset

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, *args):
        pass
