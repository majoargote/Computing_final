import pytest
from fastapi.testclient import TestClient

from api.api_fastapi import app


client = TestClient(app)


@pytest.fixture
def sample_features():
    return {
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
        "alcohol": 9.4,
    }


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body


def test_models_endpoint():
    r = client.get("/models")
    # Models endpoint may return 503 if model not loaded; accept either
    if r.status_code == 200:
        body = r.json()
        assert "required_features" in body
    else:
        assert r.status_code == 503


def test_predict_single(sample_features):
    payload = {"model": "rf", "features": sample_features}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "input_features" in body
    assert body["input_features"]["fixed_acidity"] == sample_features["fixed_acidity"]


def test_predict_batch_endpoint(sample_features):
    payload = {"model": "rf", "samples": [sample_features, sample_features]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("total_samples") == 2
    assert isinstance(body.get("results"), list)
    assert len(body.get("results")) == 2


def test_predict_file_single_and_batch(sample_features):
    # single
    payload_single = {"model": "rf", "data": sample_features}
    r1 = client.post("/predict/file", json=payload_single)
    assert r1.status_code == 200
    body1 = r1.json()
    assert "prediction" in body1

    # batch
    payload_batch = {"model": "rf", "data": [sample_features, sample_features]}
    r2 = client.post("/predict/file", json=payload_batch)
    assert r2.status_code == 200
    body2 = r2.json()
    # Depending on implementation, batch returns total_samples + results
    if "total_samples" in body2:
        assert body2["total_samples"] == 2
    else:
        # some implementations may return results directly
        assert isinstance(body2.get("results"), list)


def test_predict_multiple_json_files(tmp_path, sample_features):
    """Create multiple JSON files and POST them individually and combined.

    This test writes two JSON files to `tmp_path`: one single-sample and one batch.
    It posts each file content to `/predict/file` and finally posts a combined
    payload that merges both sources into a single batch request.
    """
    # single file
    single = {"model": "rf", "data": sample_features}
    single_file = tmp_path / "sample_single.json"
    single_file.write_text(__import__('json').dumps(single))

    # batch file
    batch = {"model": "rf", "data": [sample_features, sample_features]}
    batch_file = tmp_path / "sample_batch.json"
    batch_file.write_text(__import__('json').dumps(batch))

    # Post single file content
    with open(single_file, 'r') as f:
        payload = __import__('json').load(f)
    r1 = client.post("/predict/file", json=payload)
    assert r1.status_code == 200

    # Post batch file content
    with open(batch_file, 'r') as f:
        payload2 = __import__('json').load(f)
    r2 = client.post("/predict/file", json=payload2)
    assert r2.status_code == 200

    # Combine into one payload (merge single into batch) and post
    combined_data = []
    # payload may contain 'data' as dict or list
    def items_from(payload):
        d = payload.get('data')
        if isinstance(d, list):
            return d
        return [d]

    combined_data.extend(items_from(payload))
    combined_data.extend(items_from(payload2))

    combined_payload = {"model": "rf", "data": combined_data}
    r3 = client.post("/predict/file", json=combined_payload)
    assert r3.status_code == 200
    body3 = r3.json()
    # Expect either total_samples==len(combined_data) or results list length match
    if "total_samples" in body3:
        assert body3["total_samples"] == len(combined_data)
    else:
        assert isinstance(body3.get("results"), list)
        assert len(body3.get("results")) == len(combined_data)
