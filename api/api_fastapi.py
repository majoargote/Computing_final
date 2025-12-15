"""
FastAPI for wine classification using trained models
Compatible with uvicorn --reload
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import joblib
import os
import numpy as np
import json

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for classifying Portuguese wine using trained ML models",
    version="1.0.0"
)

# Get working directory and model paths
wd = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(wd, "models", "wine_classifier.pkl")
preprocessor_path = os.path.join(wd, "models", "preprocessor.pkl")

# Load the trained model and preprocessor
print("Loading trained model...")
model = None
preprocessor = None

try:
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded from: {preprocessor_path}")
except FileNotFoundError as e:
    print(f"Error: Model file not found. Please train the model first.")
    print(f"Missing file: {e.filename}")
except Exception as e:
    print(f"Error loading model: {e}")

# Define expected features
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid",
    "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "density", "pH",
    "sulphates", "alcohol",
]

# Pydantic models for request/response validation
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=0, le=20, description="Fixed acidity (g/dm³)")
    volatile_acidity: float = Field(..., ge=0, le=2, description="Volatile acidity (g/dm³)")
    citric_acid: float = Field(..., ge=0, le=1.5, description="Citric acid (g/dm³)")
    residual_sugar: float = Field(..., ge=0, le=20, description="Residual sugar (g/dm³)")
    chlorides: float = Field(..., ge=0, le=1, description="Chlorides (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=0, le=100, description="Free SO2 (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=0, le=300, description="Total SO2 (mg/dm³)")
    density: float = Field(..., ge=0.98, le=1.01, description="Density (g/cm³)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level")
    sulphates: float = Field(..., ge=0, le=2.5, description="Sulphates (g/dm³)")
    alcohol: float = Field(..., ge=8, le=15, description="Alcohol (% vol)")

    @validator('*', pre=True)
    def check_numeric(cls, v):
        """Ensure all values are numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Value must be numeric, got {type(v)}")
        return float(v)
    
    class Config:
        schema_extra = {
            "example": {
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


class PredictionRequest(BaseModel):
    model: str = Field(default="rf", description="Model type: 'logreg' or 'rf'")
    features: WineFeatures


class BatchPredictionRequest(BaseModel):
    model: str = Field(default="rf", description="Model type: 'logreg' or 'rf'")
    samples: List[WineFeatures]


class FilePredictionRequest(BaseModel):
    model: Optional[str] = Field(default="rf", description="Model type: 'logreg' or 'rf'")
    model_path: Optional[str] = Field(default=None, description="Path to custom model file")
    data: Union[WineFeatures, List[WineFeatures]]


class PredictionResult(BaseModel):
    class_: str = Field(..., alias="class", description="Predicted quality: poor, average, good, excellent")
    probability: Dict[str, float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Confidence of prediction")


class PredictionResponse(BaseModel):
    model_used: str
    prediction: PredictionResult
    input_features: Optional[WineFeatures] = None


@app.get("/")
async def home():
    """Home endpoint with API information"""
    return {
        "message": "Portuguese Wine Classification Prediction API",
        "version": "1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "/": "API information (this page)",
            "/predict": "POST - Make predictions using trained models",
            "/predict/batch": "POST - Batch predictions",
            "/predict/file": "POST - Predictions from JSON data with optional custom model",
            "/health": "GET - Check API health status",
            "/models": "GET - Get information about loaded models"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "logistic_regression": model_lr is not None,
            "random_forest": model_rf is not None
        }
    }


@app.get("/models")
async def models_info():
    """Get information about loaded models"""
    return {
        "models": {
            "logistic_regression": {
                "type": type(model_lr).__name__,
                "path": model_lr_path,
                "loaded": True
            },
            "random_forest": {
                "type": type(model_rf).__name__,
                "path": model_rf_path,
                "loaded": True
            }
        },
        "required_features": FEATURES
    }


def get_model(model_type: str):
    """Helper function to select the appropriate model"""
    model_type = model_type.lower()
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if model_type in ['rf', 'random_forest']:
        return model, "Random Forest Classifier (with class weights)"
    elif model_type in ['logreg', 'lr', 'logistic_regression']:
        # Add support for LR if you train one
        return model, "Logistic Regression (with class weights)"
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid model type: {model_type}",
                "valid_options": ["rf", "logreg"]
            }
        )


def make_prediction(model, features_dict: dict):
    """Helper function to make a prediction"""
    # Create feature array in correct order
    feature_values = [features_dict[f] for f in FEATURES]
    X = np.array([feature_values])
    
    # Apply preprocessing if available
    if preprocessor is not None:
        X_processed = preprocessor.transform(X)
    else:
        X_processed = X
    
    # Make prediction
    prediction_proba = model.predict_proba(X_processed)[0]
    prediction_class = model.predict(X_processed)[0]
    
    # Get class names (assuming model has classes_ attribute)
    if hasattr(model, 'classes_'):
        class_names = model.classes_
    else:
        class_names = ['poor', 'average', 'good', 'excellent']
    
    # Create probability dictionary
    prob_dict = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, prediction_proba)
    }
    
    # Determine predicted class name
    predicted_class_name = class_names[np.argmax(prediction_proba)]
    
    return {
        "class": predicted_class_name,
        "probability": prob_dict,
        "confidence": float(max(prediction_proba))
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using the trained models"""
    try:
        # Select the appropriate model
        model, model_name = get_model(request.model)
        
        # Convert Pydantic model to dict
        features_dict = request.features.model_dump()
        
        # Make prediction
        prediction = make_prediction(model, features_dict)
        
        return {
            "model_used": model_name,
            "prediction": prediction,
            "input_features": features_dict
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e)
            }
        )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple samples"""
    try:
        # Select the appropriate model
        model, model_name = get_model(request.model)
        
        # Process each sample
        results = []
        for idx, sample in enumerate(request.samples):
            try:
                features_dict = sample.model_dump()
                prediction = make_prediction(model, features_dict)
                
                results.append({
                    "sample_index": idx,
                    "prediction": prediction
                })
            except Exception as e:
                results.append({
                    "sample_index": idx,
                    "error": str(e)
                })
        
        return {
            "model_used": model_name,
            "total_samples": len(request.samples),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch prediction failed",
                "message": str(e)
            }
        )


@app.post("/predict/file")
async def predict_from_file(request: FilePredictionRequest):
    """
    Make predictions from JSON data with optional custom model path
    
    Expected JSON structure:
    {
        "model_path": "path/to/model.pkl" (optional, loads custom model),
        "model": "logreg" or "rf" (optional, uses pre-loaded model if model_path not provided),
        "data": {
            "fixed_acidity": 6.5,
            "volatile_acidity": 0.7,
            ... (single wine features)
        }
        OR
        "data": [
            {"fixed_acidity": 6.5, "volatile_acidity": 0.7, ...},
            {"fixed_acidity": 4.5, "volatile_acidity": 1.0, ...}
        ] (multiple wines)
    }
    """
    try:
        # Check if custom model path is provided
        if request.model_path:
            # Load custom model using joblib
            if not os.path.exists(request.model_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {request.model_path}"
                )
            
            try:
                model = joblib.load(request.model_path)
                model_name = f"Custom Model ({os.path.basename(request.model_path)})"
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Failed to load model",
                        "message": str(e)
                    }
                )
        else:
            # Use pre-loaded model
            model, model_name = get_model(request.model)
        
        # Check if single patient or multiple patients
        is_batch = isinstance(request.data, list)
        
        if not is_batch:
            # Single patient prediction
            features_dict = request.data.model_dump()
            prediction = make_prediction(model, features_dict)
            
            return {
                "model_used": model_name,
                "prediction": prediction,
                "input_features": features_dict
            }
        
        else:
            # Batch predictions
            results = []
            for idx, sample in enumerate(request.data):
                try:
                    features_dict = sample.model_dump()
                    prediction = make_prediction(model, features_dict)
                    
                    results.append({
                        "sample_index": idx,
                        "prediction": prediction
                    })
                except Exception as e:
                    results.append({
                        "sample_index": idx,
                        "error": str(e)
                    })
            
            return {
                "model_used": model_name,
                "total_samples": len(request.data),
                "results": results
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction from file failed",
                "message": str(e)
            }
        )
