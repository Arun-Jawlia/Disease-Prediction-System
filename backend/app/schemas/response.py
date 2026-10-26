from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    disease: str
    prediction: Literal["High Risk", "Low Risk"]
    confidence: float = Field(..., ge=0, le=100)
    model: str