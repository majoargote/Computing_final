from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    # Abstract base class for all preprocessors
    
    # ALL preprocessor classes should inherit from this and implement
    # the fit() and transform() methods.
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        # Fit preprocessor on training data
        
        #Args: X (pd.DataFrame): Training features and  y (pd.Series, optional): Training target
            
        # Returns: self: Returns self for method chaining
        pass
    
    @abstractmethod
    def transform(self, X):
        # Transform data using fitted preprocessor
        
        # Args: X (pd.DataFrame): Features to transform
            
        # Returns: pd.DataFrame: Transformed features
        
        pass
    
    def fit_transform(self, X, y=None):
        # Fit and transform in one step
        
        # Args: X (pd.DataFrame): Training features and y (pd.Series, optional): Training target
            
        # Returns: pd.DataFrame: Transformed features
        
        return self.fit(X, y).transform(X)
