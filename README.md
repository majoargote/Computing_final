# Computing_final

This repository contains a pipeline and API for predicting Portuguese wine quality.

## Quick start

- Install dependencies (recommended in a virtual environment):

```
pip install -r Requirements.txt
```

- Train and save models (saves to `api/models`):

```
python train_model.py
```

## Run the API

Before running the API you must run:

```
python -m wine_predict.pipeline
```

Start the FastAPI server (from the project root):

```
uvicorn api.api_fastapi:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`

## Example requests

PowerShell (single sample):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method POST -Body (ConvertTo-Json @{ model = 'rf'; features = @{ fixed_acidity = 7.4; volatile_acidity = 0.7; citric_acid = 0.0; residual_sugar = 1.9; chlorides = 0.076; free_sulfur_dioxide = 11.0; total_sulfur_dioxide = 34.0; density = 0.9978; pH = 3.51; sulphates = 0.56; alcohol = 9.4 } }) -ContentType 'application/json'
```

curl (single sample):

```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"model":"rf","features":{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}}'
```

## Notebooks

- `notebooks/demo_pipeline.ipynb` — pipeline demo and API usage instructions.

## Tests

Run unit tests with pytest:

```
pytest -q
```

## Where models are stored

Trained models and the preprocessor are saved to `api/models` by default. See `api/models/README.md` for details.
