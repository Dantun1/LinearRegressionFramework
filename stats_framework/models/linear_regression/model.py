from stats_framework.data import LinearRegressionDataset
from stats_framework.models.base_model import BaseModel
import numpy as np

#TODO:
# Implement LinearRegression Model that calculates coefficients(returned in fit), predicts on a np array of features.
#TODO:
# Implement .analysis() method that returns a RegressionAnalysis object.



class LinearRegression(BaseModel):
    """
    """

    @BaseModel.dataset.setter
    def dataset(self, dataset):
        super(LinearRegression, self.__class__).dataset.fset(self, dataset)

        if not isinstance(dataset, LinearRegressionDataset):
            raise TypeError("The provided dataset must be a LinearRegressionDataset")

    def fit(self):
        return 10


    def predict(self):
        return 0

    def analysis(self):
        pass




