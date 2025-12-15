from sklearn.preprocessing import StandardScaler
import pandas as pd
from wine_predict.preprocessing.base import BasePreprocessor


class StandardScalerPreprocessor(BasePreprocessor):
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
        if not self.is_fitted:
            raise RuntimeError(
                "StandardScalerPreprocessor must be fitted before calling transform()."
            )

        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(
            X_scaled,
            columns=self.feature_names,
            index=X.index
        )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
