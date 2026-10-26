from pydantic import BaseModel, Field


class LiverPredictionRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: int = Field(..., ge=0, le=1)
    total_bilirubin: float = Field(..., ge=0, le=100)
    direct_bilirubin: float = Field(..., ge=0, le=100)
    alkaline_phosphotase: float = Field(..., ge=0, le=3000)
    alamine_aminotransferase: float = Field(..., ge=0, le=3000)
    aspartate_aminotransferase: float = Field(..., ge=0, le=3000)
    total_proteins: float = Field(..., ge=0, le=15)
    albumin: float = Field(..., ge=0, le=10)
    albumin_and_globulin_ratio: float = Field(..., ge=0, le=10)