
import pandas as pd

#Class 1: Chemical Ratios 

_EPS = 0.01  # constant to avoid division by 0 

class ChemicalRatios:
    def __init__(self):
        self._feature_names = ["acidity_ratio", "sulfur_ratio"]

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()

        X_new["acidity_ratio"] = X_new["fixed_acidity"] / (X_new["volatile_acidity"] + _EPS)
        X_new["sulfur_ratio"] = X_new["free_sulfur_dioxide"] / (X_new["total_sulfur_dioxide"] + _EPS)

        return X_new

    def get_feature_names(self):
        return self._feature_names

# Class 2: Interaction Features
class InteractionFeatures:
    def __init__(self):
        self._feature_names = ["alcohol_ph", "alcohol_sulphates"]

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()

        X_new["alcohol_ph"] = X_new["alcohol"] * X_new["pH"]
        X_new["alcohol_sulphates"] = X_new["alcohol"] * X_new["sulphates"]

        return X_new

    def get_feature_names(self):
        return self._feature_names


# Class 3: Domain Features
class DomainFeatures:
    def __init__(self):
        self._feature_names = ["preservation_score", "balance_score"]

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()

        # preservation_score = alcohol * sulphates * (14 - pH)
        X_new["preservation_score"] = (
            X_new["alcohol"] * X_new["sulphates"] * (14 - X_new["pH"])
        )

        # balance_score = (citric_acid + fixed_acidity) / volatile_acidity
        
        X_new["balance_score"] = (
            (X_new["citric_acid"] + X_new["fixed_acidity"])
            / (X_new["volatile_acidity"] + _EPS)
        )

        return X_new

    def get_feature_names(self):
        return self._feature_names


# Class 4: Statistical Features
class StatisticalFeatures:
    def __init__(self):
        self._feature_names = ["alcohol_percentile", "sulphates_percentile"]

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()

        X_new["alcohol_percentile"] = X_new["alcohol"].rank(pct=True)
        X_new["sulphates_percentile"] = X_new["sulphates"].rank(pct=True)

        return X_new

    def get_feature_names(self):
        return self._feature_names


# Class 5: Polynomial Features
class PolynomialFeatures:
    def __init__(self):
        self._feature_names = ["alcohol_squared", "sulphates_squared"]

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()

        X_new["alcohol_squared"] = X_new["alcohol"] ** 2
        X_new["sulphates_squared"] = X_new["sulphates"] ** 2

        return X_new

    def get_feature_names(self):
        return self._feature_names
