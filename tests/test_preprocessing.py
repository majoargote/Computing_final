

import pytest
import pandas as pd
import numpy as np
from wine_predict.preprocessing.scalers import StandardScalerPreprocessor

#create random data and import function

@pytest.fixture
def sample_data():
    np.random.seed(123)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) * 10,
        'feature3': np.random.exponential(2, 100)
    })


@pytest.fixture
def wine_data():
    from wine_predict.data_loader import load_wine_data
    X, y = load_wine_data('data/WineQT.csv')
    return X, y

#    test StandardScaler fits and transforms correctly

def test_standard_scaler_fit_transform(sample_data):
    scaler = StandardScalerPreprocessor()
    X_scaled = scaler.fit_transform(sample_data)
    
    # Check mean ~0 and std ~1
    assert np.allclose(X_scaled.mean(), 0, atol=0.1) 
    assert np.allclose(X_scaled.std(), 1, atol=0.1) 
    assert scaler.is_fitted  # Scaler should be fitted


#same shape, missing rows or columns
def test_standard_scaler_preserves_shape(sample_data):
    scaler = StandardScalerPreprocessor()
    X_scaled = scaler.fit_transform(sample_data)
    
    assert X_scaled.shape == sample_data.shape
    assert list(X_scaled.columns) == list(sample_data.columns), "Column names should match"
    assert isinstance(X_scaled, pd.DataFrame)

#fit ONLY on train data and then applies to test correctly

def test_standard_scaler_separate_fit_transform(sample_data):
    scaler = StandardScalerPreprocessor()
    
    # Split data
    train = sample_data.iloc[:80]
    test = sample_data.iloc[80:]
    
    # Fit on train, transform both
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    
    assert train_scaled.shape == train.shape # tTrain shape preserved"
    assert test_scaled.shape == test.shape # test shape preserved"
    assert scaler.is_fitted


def test_standard_scaler_returns_self(sample_data):
    scaler = StandardScalerPreprocessor()
    result = scaler.fit(sample_data)
    
    assert result is scaler


#extra scenarios

def test_scaler_with_constant_feature():
    """Test scaler handles constant features (no variance)"""
    data = pd.DataFrame({
        'feature1': [5.0, 5.0, 5.0, 5.0],  # Constant
        'feature2': [1.0, 2.0, 3.0, 4.0]
    })
    
    scaler = StandardScalerPreprocessor()
    X_scaled = scaler.fit_transform(data)
    
    # StandardScaler will make constant features NaN or 0
    assert X_scaled.shape == data.shape