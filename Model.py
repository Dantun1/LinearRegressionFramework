from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from Data import Dataset, DataValidation

class Model(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, *args):
        pass


class LinearRegressionMixin:
    #TODO: Create methods for statistics, visualisation, hypothesis testing. All common analysis tools TBD
    ...



class SimpleLinearRegression(Model, LinearRegressionMixin):
    """
    Implements basic OLS linear regression for two-dimensional data.

    Args:
        x: Independent variable array
        y: Dependent variable array

    Raises:
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        # TODO: remove x and y from init and just pass them when training as a dataset object. User interacts with dataset objects
        # TODO: make slope and intercept properties so they can be accessed after training
        self.x = x
        self.y = y
        self._slope: Optional[float] = None
        self._intercept: Optional[float] = None

    @property
    def is_fitted(self):
        """
        Checks if the model is fitted.

        Returns:
            bool: True if the model is fitted, False otherwise
        """
        return self._slope is not None and self._intercept is not None

    def train(self):
        """
        Calculates and sets the slope and intercept of the regression line.
        """
        # TODO: Add specific validation e.g. ensuring X and Y are same length vectors (Maybe make dataValidation class)
        self._slope = self._calculate_slope()
        self._intercept = self._calculate_intercept()

    def predict(self, x):
        """
        Given a single number x, returns the predicted value y.

        Raises:
            ValueError: If the model is not fitted yet
            ValueError: If x is not a number

        Returns:
            float: The predicted value of y
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet, call .train() first")

        if  not isinstance(x, (int, float)):
            raise ValueError("x must be a number")


        return self._intercept + self._slope * x


    def _calculate_slope(self):
        """
        Calculates the least squares regression slope coefficient

        Uses the formula slope = Cov(xy)/Var(x) = Sxy / Sxx

        Returns:
            float: The slope coefficient
        """
        Sxy = np.sum(self.x * self.y) - np.sum(self.x) * np.sum(self.y) / len(self.x)
        Sxx = np.sum(self.x ** 2) - np.sum(self.x) ** 2 / len(self.x)
        return Sxy / Sxx

    def _calculate_intercept(self):
        """
        Calculates the intercept of the regression line using the value of the slope if set,

        Raises:
            ValueError: If the slope is not set

        Returns:
            float: The intercept of the regression line
        """
        if self._slope is None:
            raise ValueError("Slope is not calculated, call ._calculate_slope() first or .train() ")
        return np.mean(self.y) - self._calculate_slope() * np.mean(self.x)

    def __str__(self):
        message = ""
        if not self.is_fitted:
            message += "Model is not fitted yet, call .train() first"
        else:
            if self._slope >= 0:
                message += f"y = {self._intercept:.2f} + {self._slope:.2f}x"
            elif self._slope < 0:
                message += f"y = {self._intercept:.2f} - {-self._slope:.2f}x"


        return message


if __name__ == "__main__":
    x = np.random.rand(5) * 10
    y = np.array([2, 4, 6, 8, 10])
    model = SimpleLinearRegression(x, y)
    print(model)
    model.train()
    print(model)

