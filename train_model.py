"""
Train Wine Quality Model
Run this script to train the model and save it for the API
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wine_predict.pipeline import WineQualityPipeline
import argparse


def main(tune: bool = False):
    print("\n" + "="*70)
    print("TRAINING WINE QUALITY CLASSIFICATION MODEL")
    print("="*70)
    
    # Initialize pipeline
    pipeline = WineQualityPipeline(random_state=42)
    
    if tune:
        print("\n>>> Running hyperparameter tuning (Random Forest)")
        # Example parameter grid (GridSearch)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        metrics = pipeline.run_full_pipeline(
            data_path='data/WineQT.csv',
            model_type='random_forest',
            save_model=True,
            model_dir='api/models',
            tune=True,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy'
        )
    else:
        # Train and save Random Forest model with example params
        print("\n>>> Training Random Forest Classifier")
        metrics = pipeline.run_full_pipeline(
            data_path='data/WineQT.csv',
            model_type='random_forest',
            save_model=True,
            model_dir='api/models',
            n_estimators=100,
            max_depth=10,
            min_samples_split=5
        )
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nModel saved to: api/models/")
    print("\nYou can now start the API server with:")
    print("  uvicorn api.api_fastapi:app --reload")
    print("\nOr:")
    print("  python -m uvicorn api.api_fastapi:app --reload")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train wine quality model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning before training')
    args = parser.parse_args()
    main(tune=args.tune)
