# Model Directory

This directory stores trained machine learning models for the Wine Quality Prediction API.

## Files
- `wine_classifier.pkl` - Trained classification model (Random Forest or Logistic Regression)
- `preprocessor.pkl` - Fitted StandardScaler preprocessor
- `feature_names.txt` - List of feature names after feature engineering

## Training Models

To train and save models, run:

```bash
python train_model.py
```

Or use the WineQualityPipeline class in your code:

```python
from wine_predict.pipeline import WineQualityPipeline

pipeline = WineQualityPipeline()
pipeline.run_full_pipeline(
    data_path='data/WineQT.csv',
    model_type='random_forest',
    save_model=True,
    model_dir='api/models'
)
```

## Using the API

Once models are saved in this directory you can start the API server and make predictions:

Start server (project root):

```
uvicorn api.api_fastapi:app --host 127.0.0.1 --port 8000 --reload
```

Use `/predict` for single-sample predictions, `/predict/batch` for multiple samples, and `/predict/file` for JSON file uploads. See the interactive documentation at `http://127.0.0.1:8000/docs` for payload examples.

There is also a client script to run batch predictions and save results: `scripts/batch_predict_api.py`.
