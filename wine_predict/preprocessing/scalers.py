from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from .base import BasePreprocessor


class StandardScalerPreprocessor:
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X, y=None):

        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(
            X_scaled,
            columns=self.feature_names,
            index=X.index
        )


class RobustScalerPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()
        self.scaler = RobustScaler()
        self.feature_names = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(
            X_scaled,
            columns=self.feature_names,
            index=X.index
        )

