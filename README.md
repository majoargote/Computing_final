# Computing_final

Authors:

- Mariajosé Argote
- María Victoria Suriel
- Elvis Casco

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

curl - using git (single sample):

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

## **Extensibility / Contribution Guidelines**

This project is designed to be extensible so you can add new preprocessors, feature transformers, models, and metrics with minimal changes. Follow the conventions below to keep the codebase consistent and testable.

- **Design principles**: Keep components small and focused, provide a clear public API (class methods like `fit`, `transform`, `compute`, `get_feature_names`, `save`, `load`), and add unit tests for every new piece.

- **Repository layout (relevant folders)**:
	- `wine_predict/preprocessing/` — preprocessors (scalers, imputers, pipelines)
	- `wine_predict/features/` — feature engineering transformers
	- `wine_predict/` — pipeline orchestration (`pipeline.py`) and utilities
	- `api/models/` — saved artifacts (models, preprocessor, feature names)

- **Adding a new preprocessor**:
	- Create a new module under `wine_predict/preprocessing/`, e.g. `my_scaler.py`.
	- Implement a class that follows the existing `BasePreprocessor` interface (`fit`, `transform`, `fit_transform`, `save`/`load` optional).
	- Register or instantiate it in `wine_predict/pipeline.py` (replace or add to `self.preprocessor`). Update `save_pipeline`/`load_pipeline` if custom serialization is required.

- **Adding a new feature transformer**:
	- Add a new file in `wine_predict/features/` (e.g. `my_features.py`).
	- Implement a transformer with these methods: `compute(self, X: pd.DataFrame) -> pd.DataFrame` and `get_feature_names(self) -> List[str]`.
	- Add the class to `WineQualityPipeline().feature_engineers` list in `pipeline.py` to include it in the standard engineering step.

- **Adding a new model**:
	- Implement a training wrapper inside `wine_predict/pipeline.py` or a new module under `wine_predict/models/` (create the folder if needed).
	- The training method should accept `X_train, y_train` and return a fitted estimator exposing `predict` and `predict_proba`.
	- Update the `train_model` factory in `WineQualityPipeline` to support the new `model_type` string and default hyperparameters.
	- Ensure the trained model is saved to `api/models/wine_classifier.pkl` (or an alternative name) via `joblib.dump` so the API can load it.

- **Adding new evaluation metrics**:
	- Add metric functions in `wine_predict/metrics.py` (create the file), e.g. `def brier_score(y_true, y_prob): ...`.
	- In `pipeline.evaluate_model()`, compute and include these metrics in the returned metrics dict. Add unit tests for each metric.

- **Testing & CI**:
	- Add unit tests under `tests/` that exercise new preprocessors, feature transformers, and models.
	- Use small, deterministic inputs (pandas DataFrames) so tests run quickly.
	- Run `pytest -q` locally and ensure coverage for new code paths.

- **API integration**:
	- If your change affects input feature names or expected JSON schema, update `api/api_fastapi.py` to validate and document the change.
	- Keep backward compatibility where possible; if not, bump the API contract and document the migration in this README.
