import pytest
import json
from src.app import app


@pytest.fixture
def client():
    """Create a test client for the app."""
    app.testing = True
    return app.test_client()


def test_home(client):
    """Test the home route."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sentiment Analysis API is running!" in response.data


def test_predict_positive(client):
    """Test positive sentiment."""
    response = client.post(
        "/predict",
        data=json.dumps({"text": "I love this movie!"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert json.loads(response.data)["sentiment"] in ["positive", "negative"]


def test_predict_negative(client):
    """Test negative sentiment."""
    response = client.post(
        "/predict",
        data=json.dumps({"text": "This is the worst film ever!"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert json.loads(response.data)["sentiment"] in ["positive", "negative"]


def test_predict_empty(client):
    """Test empty input."""
    response = client.post(
        "/predict",
        data=json.dumps({"text": ""}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "error" in json.loads(response.data)
