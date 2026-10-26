import pandas as pd


def make_prediction(model, features: dict):
    data = pd.DataFrame([features])

    prediction = model.predict(data)[0]

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]

    return prediction, probabilities