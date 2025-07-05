from typing import Optional

from stats_framework.data import LinearRegressionDataset
from stats_framework.models.base_model import BaseModel
from stats_framework.models.linear_regression.analysis import RegressionAnalysis
from stats_framework.shared_utilities.exceptions.model_errors import NotFittedError


#TODO:
# Implement LinearRegression Model that calculates coefficients(returned in fit), predicts on a np array of features.

class LinearRegression(BaseModel):
    """
    """
    def __init__(self, dataset: LinearRegressionDataset):
        super(LinearRegression, self).__init__(dataset)
        self.coefficients: Optional[list[float]] = None

    @BaseModel.dataset.setter
    def dataset(self, dataset):
        super(LinearRegression, self.__class__).dataset.fset(self, dataset)

        if not isinstance(dataset, LinearRegressionDataset):
            raise TypeError("The provided dataset must be a LinearRegressionDataset")

    def fit(self) -> None:
        return

    def predict(self):
        return 0

    @property
    def is_fitted(self):
        return self.coefficients is not None


    def analysis(self, **kwargs) -> RegressionAnalysis:
        """
        This method tries to return a RegressionAnalysis object giving access to statistical metrics and visualizations.

        The user may input a different LinearRegressionDataset to the one the model has been trained on,
        if getting analysis fails due to the invalid type of inputted dataset,
        return the analysis on the pre-validated training dataset.

        Raises:
            NotFittedError if the coefficients are not fitted when RegressionAnalysis object is instantiated.


        Return a regression analysis object
        """

        try:
            if kwargs.get("dataset") is None:
                regression_analysis = RegressionAnalysis(self)
            else:
                try:
                    regression_analysis = RegressionAnalysis(self, dataset=kwargs.get("dataset"))
                except TypeError as e:
                    print(f"Error analysing model with inputted data:{e}")
                    regression_analysis = RegressionAnalysis(self)

        except NotFittedError as e:
            raise NotFittedError(e)

        return regression_analysis





