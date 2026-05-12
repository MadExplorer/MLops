from fastapi import FastAPI
from .schemas import WineFeatures
from .predictor import predict
from .database import engine, SessionLocal
from .models import Base, Prediction

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Wine Quality API"}

@app.post("/predict")
def make_prediction(features: WineFeatures):

    values = list(features.dict().values())

    result = predict(values)

    db = SessionLocal()

    record = Prediction(
        alcohol=features.alcohol,
        prediction=result
    )

    db.add(record)
    db.commit()

    return {"prediction": result}