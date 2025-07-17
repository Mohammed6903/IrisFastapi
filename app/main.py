from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ml_model/iris_random_forest_model.joblib")

model = joblib.load(model_path)
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
    features = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }])

    prediction = model.predict(features)[0]
    species = iris.target_names[prediction]

    return {
        "prediction": int(prediction),
        "species": species
    }
