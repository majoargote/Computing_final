from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):
    # Abstract base class for all preprocessors

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y=None):
        # Fit preprocessor on training data
        raise NotImplementedError()

    @abstractmethod
    def transform(self, X):
        # Transform data using fitted preprocessor
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        # Fit and transform in one step
        self.fit(X, y)
        return self.transform(X)