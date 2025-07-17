from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris

model = joblib.load("iris_random_forest_model.joblib")
iris = load_iris()

app = FastAPI(title="Iris Classifier API", description="Predict Iris species", version="1.0")


# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "Welcome to the Iris Prediction API"}


@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]
    species = iris.target_names[prediction]

    return {
        "prediction": int(prediction),
        "species": species
    }
