from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService


model_service = ModelService()

prediction_service = PredictionService(
    model_service
)