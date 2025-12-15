# Wine Quality Prediction API

FastAPI-based REST API for predicting wine quality using machine learning.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r Requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train_model.py
   ```
   
   This will:
   - Load the wine quality dataset
   - Engineer features (chemical ratios, interactions, domain features, etc.)
   - Train a Random Forest classifier
   - Save the model to `api/models/`

3. **Start the API server:**
   ```bash
   uvicorn api.api_fastapi:app --reload
   ```
   
   The API will be available at: `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Interactive API docs:** `http://localhost:8000/docs`
- **Alternative docs:** `http://localhost:8000/redoc`
- **OpenAPI schema:** `http://localhost:8000/openapi.json`

## API Endpoints

### `GET /`
API information and available endpoints

### `GET /health`
Health check - verify the API and model are loaded

### `GET /models`
Information about the loaded model and required features

### `POST /predict`
Make a single prediction

**Request body:**
```json
{
  "model": "rf",
  "features": {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }
}
```

**Response:**
```json
{
  "model_used": "RandomForestClassifier",
  "prediction": {
    "class": "average",
    "probability": {
      "poor": 0.05,
      "average": 0.85,
      "good": 0.10
    },
    "confidence": 0.85
  },
  "input_features": { ... }
}
```

### `POST /predict/batch`
Make predictions for multiple samples

**Request body:**
```json
{
  "model": "rf",
  "samples": [
    {
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.7,
      ...
    },
    {
      "fixed_acidity": 8.1,
      "volatile_acidity": 0.6,
      ...
    }
  ]
}
```

### `POST /predict/file`
Make predictions from JSON data with optional custom model

## Features

The API automatically performs feature engineering on input data:
- **Chemical Ratios:** acidity_ratio, sulfur_ratio
- **Interaction Features:** alcohol_ph, alcohol_sulphates
- **Domain Features:** preservation_score, balance_score
- **Statistical Features:** alcohol_percentile, sulphates_percentile
- **Polynomial Features:** alcohol_squared, sulphates_squared

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "model": "rf",
    "features": {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rf",
    "features": {
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.7,
      "citric_acid": 0.0,
      "residual_sugar": 1.9,
      "chlorides": 0.076,
      "free_sulfur_dioxide": 11.0,
      "total_sulfur_dioxide": 34.0,
      "density": 0.9978,
      "pH": 3.51,
      "sulphates": 0.56,
      "alcohol": 9.4
    }
  }'
```

## Project Structure

```
Computing_final/
├── api/
│   ├── api_fastapi.py      # FastAPI application
│   └── models/             # Trained models directory
├── wine_predict/
│   ├── data_loader.py      # Data loading utilities
│   ├── pipeline.py         # ML pipeline
│   ├── features/           # Feature engineering
│   └── preprocessing/      # Data preprocessing
├── data/
│   └── WineQT.csv         # Wine quality dataset
├── train_model.py         # Script to train the model
└── Requirements.txt       # Python dependencies
```

## Development

To run the API in development mode with auto-reload:
```bash
uvicorn api.api_fastapi:app --reload --host 0.0.0.0 --port 8000
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=wine_predict --cov-report=html
```
