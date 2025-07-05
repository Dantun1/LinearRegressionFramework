import numpy as np
# from stats_framework.models.linear_regression.model import LinearRegression
from stats_framework.data.linear_regression_dataset import LinearRegressionDataset
from stats_framework.shared_utilities.exceptions.model_errors import NotFittedError

#TODO:
# -Validation of dimensions of model coefficients and dataset
# -Implement r_squared method

class RegressionAnalysis:

    def __init__(self, model, **kwargs):
        self.model = model

        dataset = kwargs.get("dataset")
        if dataset is None:
            print("No dataset provided, using training dataset")
            dataset = self.model.dataset
        elif not isinstance(dataset, LinearRegressionDataset):
            raise TypeError("dataset must be of type LinearRegressionDataset")

        self.dataset = dataset
        self.predictions = self._get_predictions()

    def r_squared(self):
        predictions = self.predictions
        response = self.dataset.response

        tss = sum((response - response.mean()) ** 2)
        rss = sum((response - predictions)**2)

        return (tss - rss)/ rss



    def _get_predictions(self) -> np.ndarray:
        features_matrix = self.dataset.features.to_numpy()

        if self.model.coefficients is None:
            raise NotFittedError("model must be fitted before predictions can be made")

        coefficients = np.array(self.model.coefficients)

        predictions = features_matrix * coefficients

        return predictions


