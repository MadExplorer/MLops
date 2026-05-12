import joblib
import numpy as np

model = joblib.load("app/model/wine_model.pkl")

def predict(features):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    return int(prediction[0])