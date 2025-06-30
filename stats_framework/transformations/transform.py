from abc import ABC, abstractmethod

class _Transform(ABC):

    def __init__(self, dataset):
        self.dataset = dataset


