from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Prediction API"}


def test_predict_setosa():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["prediction"] == 0
    assert json_data["species"] == "setosa"


def test_predict_versicolor():
    response = client.post("/predict", json={
        "sepal_length": 6.0,
        "sepal_width": 2.9,
        "petal_length": 4.5,
        "petal_width": 1.5
    })
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["prediction"] == 1
    assert json_data["species"] == "versicolor"


def test_predict_virginica():
    response = client.post("/predict", json={
        "sepal_length": 6.9,
        "sepal_width": 3.1,
        "petal_length": 5.4,
        "petal_width": 2.1
    })
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["prediction"] == 2
    assert json_data["species"] == "virginica"


def test_predict_invalid_input_missing_field():
    response = client.post("/predict", json={
        "sepal_length": 6.9,
        "sepal_width": 3.1,
        "petal_length": 5.4
        # missing petal_width
    })
    assert response.status_code == 422


def test_predict_invalid_input_wrong_type():
    response = client.post("/predict", json={
        "sepal_length": "big",
        "sepal_width": 3.1,
        "petal_length": 5.4,
        "petal_width": 2.1
    })
    assert response.status_code == 422
