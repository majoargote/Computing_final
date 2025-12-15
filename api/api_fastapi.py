"""
FastAPI for wine classification using trained models
Compatible with uvicorn --reload
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union
import joblib
import os
import sys
import numpy as np
import pandas as pd
import json

# Add parent directory to path to import wine_predict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wine_predict.pipeline import WineQualityPipeline

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for classifying Portuguese wine using trained ML models",
    version="1.0.0"
)

# Initialize pipeline
pipeline = None
wd = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(wd, "models")

# Load the trained model and preprocessor
print("Loading trained pipeline...")
try:
    if os.path.exists(os.path.join(model_dir, "wine_classifier.pkl")):
        pipeline = WineQualityPipeline()
        pipeline.load_pipeline(model_dir)
        print("Pipeline loaded successfully!")
    else:
        print("No trained model found. Please train the model first by running:")
        print("  python wine_predict/pipeline.py")
except Exception as e:
    print(f"Error loading pipeline: {e}")

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

    @field_validator('*', mode='before')
    def check_numeric(cls, v):
        """Ensure all values are numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Value must be numeric, got {type(v)}")
        return float(v)
    
    class Config:
        json_schema_extra = {
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
        "pipeline_loaded": pipeline is not None,
        "model_loaded": pipeline is not None and pipeline.model is not None
    }


@app.get("/models")
async def models_info():
    """Get information about loaded models"""
    if pipeline is None or pipeline.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model": {
            "type": type(pipeline.model).__name__,
            "loaded": True,
            "features": pipeline.feature_names if hasattr(pipeline, 'feature_names') else None
        },
        "required_features": FEATURES,
        "feature_engineering": "ChemicalRatios, InteractionFeatures, DomainFeatures, StatisticalFeatures, PolynomialFeatures"
    }


def check_pipeline():
    """Check if pipeline is loaded"""
    if pipeline is None or pipeline.model is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Please train the model first by running: python wine_predict/pipeline.py"
        )
    return pipeline


def make_prediction(features_dict: dict):
    """Helper function to make a prediction using the pipeline"""
    # Create DataFrame with features in correct order
    X = pd.DataFrame([features_dict], columns=FEATURES)
    
    # Use pipeline to predict (handles feature engineering and preprocessing)
    predictions, probabilities = pipeline.predict(X)
    
    # Get class names
    class_names = pipeline.model.classes_
    
    # Create probability dictionary
    prob_dict = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities[0])
    }
    
    # Determine predicted class name
    predicted_class_name = predictions[0]
    
    return {
        "class": predicted_class_name,
        "probability": prob_dict,
        "confidence": float(max(probabilities[0]))
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using the trained model"""
    try:
        # Check pipeline is loaded
        check_pipeline()
        
        # Convert Pydantic model to dict
        features_dict = request.features.model_dump()
        
        # Make prediction
        prediction = make_prediction(features_dict)
        
        model_name = type(pipeline.model).__name__
        
        return {
            "model_used": model_name,
            "prediction": prediction,
            "input_features": features_dict
        }
    
    except HTTPException:
        raise
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
        # Check pipeline is loaded
        check_pipeline()
        
        model_name = type(pipeline.model).__name__
        
        # Process each sample
        results = []
        for idx, sample in enumerate(request.samples):
            try:
                features_dict = sample.model_dump()
                prediction = make_prediction(features_dict)
                
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
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch prediction failed",
                "message": str(e)
            }
        )


@app.post("/predict/file")
async def predict_from_file(req: Request, file: Optional[UploadFile] = File(None)):
    """
    Make predictions from JSON data with optional custom model path
    
    Expected JSON structure:
    {
        "model_path": "path/to/model.pkl" (optional, loads custom model),
        "model": "unused" (kept for backwards compatibility),
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
        # Determine source of payload: uploaded file or JSON body
        payload = None

        # If a file was uploaded via the docs UI, read and parse it
        if file is not None:
            try:
                content = await file.read()
                parsed = json.loads(content)
            except Exception as e:
                raise HTTPException(status_code=400, detail={"error": "Failed to read/parse uploaded file", "message": str(e)})

            # Accept either raw data (list or single object) or a wrapper dict with 'data', 'model', 'model_path'
            if isinstance(parsed, dict) and ("data" in parsed or "model_path" in parsed or "model" in parsed):
                payload = parsed
            else:
                # If user uploaded a plain list or single object, wrap it into the expected shape
                payload = {"data": parsed}

        else:
            # Try to read JSON body directly (handles application/json requests)
            try:
                body = await req.json()
            except Exception:
                body = None

            if not body:
                raise HTTPException(status_code=400, detail="No input provided. Send JSON body or upload a JSON file.")

            payload = body

        # At this point payload should be a dict possibly containing 'model_path', 'model', and 'data'
        active_pipeline = pipeline
        model_name = type(pipeline.model).__name__ if pipeline and pipeline.model is not None else "No model loaded"

        # If a model_path is provided, attempt to load model from that directory
        model_path = payload.get("model_path") if isinstance(payload, dict) else None
        if model_path:
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
            try:
                custom_pipeline = WineQualityPipeline()
                custom_pipeline.load_pipeline(os.path.dirname(model_path))
                active_pipeline = custom_pipeline
                model_name = f"Custom Model ({os.path.basename(model_path)})"
            except Exception as e:
                raise HTTPException(status_code=500, detail={"error": "Failed to load custom model", "message": str(e)})
        else:
            # Use pre-loaded pipeline
            check_pipeline()

        # Extract data from payload
        data_field = payload.get("data") if isinstance(payload, dict) else None
        if data_field is None:
            raise HTTPException(status_code=400, detail="Request payload missing 'data' field (single object or list of objects) or upload a JSON file containing data.")

        # Determine single vs batch
        is_batch = isinstance(data_field, list)

        if not is_batch:
            # single sample
            features_dict = data_field
            try:
                X = pd.DataFrame([features_dict], columns=FEATURES)
            except Exception as e:
                raise HTTPException(status_code=400, detail={"error": "Invalid feature structure", "message": str(e)})

            try:
                predictions, probabilities = active_pipeline.predict(X)
            except Exception as e:
                raise HTTPException(status_code=500, detail={"error": "Prediction failed", "message": str(e)})

            class_names = active_pipeline.model.classes_
            prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities[0])}
            prediction = {"class": predictions[0], "probability": prob_dict, "confidence": float(max(probabilities[0]))}

            return {"model_used": model_name, "prediction": prediction, "input_features": features_dict}

        else:
            # batch
            results = []
            for idx, sample in enumerate(data_field):
                try:
                    features_dict = sample
                    X = pd.DataFrame([features_dict], columns=FEATURES)
                    predictions, probabilities = active_pipeline.predict(X)
                    class_names = active_pipeline.model.classes_
                    prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities[0])}
                    prediction = {"class": predictions[0], "probability": prob_dict, "confidence": float(max(probabilities[0]))}
                    results.append({"sample_index": idx, "prediction": prediction})
                except Exception as e:
                    results.append({"sample_index": idx, "error": str(e)})

            return {"model_used": model_name, "total_samples": len(data_field), "results": results}
    
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
