import pytest
import pandas as pd
import numpy as np

from wine_predict.data_loader import load_wine_data
from wine_predict.features.features import (
    ChemicalRatios,
    InteractionFeatures,
    DomainFeatures,
    StatisticalFeatures,
    PolynomialFeatures,
)

_EPS = 0.01


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def wine_X():
    X, _ = load_wine_data("data/WineQT.csv")
    return X


@pytest.fixture
def small_df():
    # Small DataFrame to prove calculation
    return pd.DataFrame(
        {
            "fixed_acidity": [10.0, 20.0],
            "volatile_acidity": [2.0, 4.0],
            "free_sulfur_dioxide": [10.0, 20.0],
            "total_sulfur_dioxide": [40.0, 80.0],
            "alcohol": [10.0, 12.0],
            "pH": [3.0, 4.0],
            "sulphates": [0.5, 0.8],
            "citric_acid": [0.2, 0.3],
        }
    )


# =========================================================
# Class 1: ChemicalRatios
# =========================================================
def test_chemical_ratios_adds_columns(wine_X):
    X2 = ChemicalRatios().compute(wine_X)
    assert "acidity_ratio" in X2.columns
    assert "sulfur_ratio" in X2.columns


def test_chemical_ratios_preserves_rows(wine_X):
    X2 = ChemicalRatios().compute(wine_X)
    assert len(X2) == len(wine_X)


def test_chemical_ratios_calculation_accuracy(small_df):
    X2 = ChemicalRatios().compute(small_df)

    expected_acidity_0 = 10.0 / (2.0 + _EPS)
    expected_sulfur_0 = 10.0 / (40.0 + _EPS)

    assert np.isclose(X2.loc[0, "acidity_ratio"], expected_acidity_0)
    assert np.isclose(X2.loc[0, "sulfur_ratio"], expected_sulfur_0)


# =========================================================
# Class 2: InteractionFeatures
# =========================================================
def test_interaction_features_adds_columns(wine_X):
    X2 = InteractionFeatures().compute(wine_X)
    assert "alcohol_ph" in X2.columns
    assert "alcohol_sulphates" in X2.columns


def test_interaction_features_preserves_rows(wine_X):
    X2 = InteractionFeatures().compute(wine_X)
    assert len(X2) == len(wine_X)


def test_interaction_features_calculation_accuracy(small_df):
    X2 = InteractionFeatures().compute(small_df)

    assert np.isclose(X2.loc[0, "alcohol_ph"], 10.0 * 3.0)
    assert np.isclose(X2.loc[0, "alcohol_sulphates"], 10.0 * 0.5)


# =========================================================
# Class 3: DomainFeatures
# =========================================================
def test_domain_features_adds_columns(wine_X):
    X2 = DomainFeatures().compute(wine_X)
    assert "preservation_score" in X2.columns
    assert "balance_score" in X2.columns


def test_domain_features_preserves_rows(wine_X):
    X2 = DomainFeatures().compute(wine_X)
    assert len(X2) == len(wine_X)


def test_domain_features_calculation_accuracy(small_df):
    X2 = DomainFeatures().compute(small_df)

    expected_pres_0 = 10.0 * 0.5 * (14 - 3.0)
    expected_balance_0 = (0.2 + 10.0) / (2.0 + _EPS)

    assert np.isclose(X2.loc[0, "preservation_score"], expected_pres_0)
    assert np.isclose(X2.loc[0, "balance_score"], expected_balance_0)


# =========================================================
# Class 4: StatisticalFeatures
# =========================================================
def test_statistical_features_adds_columns(wine_X):
    X2 = StatisticalFeatures().compute(wine_X)
    assert "alcohol_percentile" in X2.columns
    assert "sulphates_percentile" in X2.columns


def test_statistical_features_values_between_0_and_1(wine_X):
    X2 = StatisticalFeatures().compute(wine_X)
    assert X2["alcohol_percentile"].between(0, 1).all()
    assert X2["sulphates_percentile"].between(0, 1).all()


def test_statistical_features_percentile_accuracy_simple():
    df = pd.DataFrame({"alcohol": [10.0, 20.0, 30.0], "sulphates": [1.0, 2.0, 3.0]})
    X2 = StatisticalFeatures().compute(df)

    # rank(pct=True) con valores únicos -> [1/3, 2/3, 1]
    assert np.isclose(X2.loc[0, "alcohol_percentile"], 1 / 3)
    assert np.isclose(X2.loc[1, "alcohol_percentile"], 2 / 3)
    assert np.isclose(X2.loc[2, "alcohol_percentile"], 1.0)


# =========================================================
# Class 5:PolynomialFeatures
# =========================================================
def test_polynomial_features_adds_columns(wine_X):
    X2 = PolynomialFeatures().compute(wine_X)
    assert "alcohol_squared" in X2.columns
    assert "sulphates_squared" in X2.columns


def test_polynomial_features_preserves_rows(wine_X):
    X2 = PolynomialFeatures().compute(wine_X)
    assert len(X2) == len(wine_X)


def test_polynomial_features_calculation_accuracy(small_df):
    X2 = PolynomialFeatures().compute(small_df)
    assert np.isclose(X2.loc[0, "alcohol_squared"], 10.0 ** 2)
    assert np.isclose(X2.loc[0, "sulphates_squared"], 0.5 ** 2)
