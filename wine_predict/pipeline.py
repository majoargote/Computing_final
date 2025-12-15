"""
Wine Quality Prediction Pipeline
Handles data loading, feature engineering, preprocessing, training, and model saving
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import os
from typing import Tuple, Optional

from wine_predict.data_loader import load_wine_data
from wine_predict.preprocessing.scalers import StandardScalerPreprocessor
from wine_predict.features.features import (
    ChemicalRatios, 
    InteractionFeatures, 
    DomainFeatures,
    StatisticalFeatures,
    PolynomialFeatures
)


class WineQualityPipeline:
    """Complete pipeline for wine quality prediction"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = StandardScalerPreprocessor()
        
        # Feature engineering transformers
        self.feature_engineers = [
            ChemicalRatios(),
            InteractionFeatures(),
            DomainFeatures(),
            StatisticalFeatures(),
            PolynomialFeatures()
        ]
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self, data_path: str = 'data/WineQT.csv') -> Tuple[pd.DataFrame, pd.Series]:
        """Load wine quality data"""
        print("Loading data...")
        X, y = load_wine_data(data_path)
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        print("\nEngineering features...")
        X_engineered = X.copy()
        
        for engineer in self.feature_engineers:
            X_engineered = engineer.compute(X_engineered)
            new_features = engineer.get_feature_names()
            print(f"  Added {len(new_features)} features: {', '.join(new_features)}")
        
        print(f"Total features after engineering: {X_engineered.shape[1]}")
        return X_engineered
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def preprocess_data(self):
        """Fit preprocessor on train data and transform both train and test"""
        print("\nPreprocessing data...")
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        self.feature_names = self.X_train.columns.tolist()
        print(f"Preprocessor fitted and applied")
    
    def train_model(self, model_type: str = 'random_forest', **model_params):
        """Train a classification model"""
        print(f"\nTraining {model_type} model...")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                min_samples_split=model_params.get('min_samples_split', 5),
                class_weight=class_weight_dict,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=model_params.get('max_iter', 1000),
                class_weight=class_weight_dict,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(self.X_train, self.y_train)
        print(f"Model training complete")
        
        return self.model

    def tune_hyperparameters(
        self,
        model_type: str = 'random_forest',
        param_grid: Optional[dict] = None,
        cv: int = 5,
        n_iter: Optional[int] = None,
        scoring: str = 'accuracy',
        random_state: Optional[int] = None,
    ) -> dict:
        """Perform hyperparameter search (Grid or Randomized) and set the best estimator as `self.model`.

        - If `n_iter` is provided, uses RandomizedSearchCV with `n_iter` iterations.
        - Otherwise uses GridSearchCV over `param_grid`.
        Returns a dict with `best_params` and `best_score`.
        """
        print(f"\nTuning hyperparameters for {model_type} (cv={cv}, scoring={scoring})...")

        if model_type == 'random_forest':
            base_estimator = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
        elif model_type == 'logistic_regression':
            base_estimator = LogisticRegression(max_iter=2000, random_state=self.random_state)
            if param_grid is None:
                param_grid = {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l2']
                }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        search = None
        if n_iter is not None:
            print(f"Using RandomizedSearchCV with n_iter={n_iter}")
            search = RandomizedSearchCV(
                base_estimator, param_distributions=param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, random_state=random_state or self.random_state, n_jobs=-1
            )
        else:
            print("Using GridSearchCV")
            search = GridSearchCV(base_estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)

        # Fit search on preprocessed training data
        print("Fitting hyperparameter search on training data...")
        search.fit(self.X_train, self.y_train)

        print(f"Best score: {search.best_score_}")
        print(f"Best params: {search.best_params_}")

        # Set the best estimator
        self.model = search.best_estimator_

        return {'best_params': search.best_params_, 'best_score': search.best_score_}
    
    def evaluate_model(self) -> dict:
        """Evaluate model performance on test set"""
        print("\nEvaluating model...")
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
    
    def save_pipeline(self, model_dir: str = 'api/models'):
        """Save the trained model and preprocessor"""
        print(f"\nSaving pipeline to {model_dir}...")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'wine_classifier.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")
        
        # Save feature names
        feature_names_path = os.path.join(model_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        print(f"Feature names saved to: {feature_names_path}")
    
    def load_pipeline(self, model_dir: str = 'api/models'):
        """Load a saved model and preprocessor"""
        print(f"\nLoading pipeline from {model_dir}...")
        
        # Load model
        model_path = os.path.join(model_dir, 'wine_classifier.pkl')
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        self.preprocessor = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded from: {preprocessor_path}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        # Engineer features
        X_engineered = self.engineer_features(X)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X_engineered)
        
        # Predict
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        return predictions, probabilities
    
    def run_full_pipeline(
        self, 
        data_path: str = 'data/WineQT.csv',
        model_type: str = 'random_forest',
        save_model: bool = True,
        model_dir: str = 'api/models',
        tune: bool = False,
        param_grid: Optional[dict] = None,
        n_iter: Optional[int] = None,
        cv: int = 5,
        scoring: str = 'accuracy',
        **model_params
    ) -> dict:
        """Run the complete pipeline from data loading to model saving"""
        print("="*60)
        print("WINE QUALITY PREDICTION PIPELINE")
        print("="*60)
        
        # Load data
        X, y = self.load_data(data_path)
        
        # Engineer features
        X_engineered = self.engineer_features(X)
        
        # Split data
        self.split_data(X_engineered, y)
        
        # Preprocess
        self.preprocess_data()
        # Train (or tune) model
        if tune:
            tuning_result = self.tune_hyperparameters(
                model_type=model_type,
                param_grid=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring
            )
            print(f"Tuning result: {tuning_result}")
        else:
            self.train_model(model_type, **model_params)

        # Evaluate
        metrics = self.evaluate_model()
        
        # Save pipeline
        if save_model:
            self.save_pipeline(model_dir)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return metrics


if __name__ == "__main__":
    # Example usage
    pipeline = WineQualityPipeline(random_state=42)
    
    # Train Random Forest
    print("\n>>> Training Random Forest Classifier")
    metrics_rf = pipeline.run_full_pipeline(
        data_path='data/WineQT.csv',
        model_type='random_forest',
        save_model=True,
        model_dir='api/models',
        n_estimators=100,
        max_depth=10,
        min_samples_split=5
    )
