from app.ml.predictor import make_prediction
from app.services.model_service import ModelService


class PredictionService:

    def __init__(
        self,
        model_service: ModelService,
    ):
        self.model_service = model_service

    def predict(
        self,
        disease: str,
        features: dict,
    ):

        model = self.model_service.get_model(
            disease
        )

        metadata = self.model_service.get_metadata(
            disease
        )

        prediction, probabilities = make_prediction(
            model,
            features,
        )

        prediction = int(prediction)

        if prediction == 1:
            risk = "High Risk"
        else:
            risk = "Low Risk"

        confidence = None

        if probabilities is not None:
            confidence = float(
                max(probabilities) * 100
            )

        model_name = metadata.get(
            "model_name",
            "unknown",
        )

        return {
            "disease": disease,
            "prediction": risk,
            "confidence": (
                round(confidence, 2)
                if confidence is not None
                else None
            ),
            "model": model_name,
        }