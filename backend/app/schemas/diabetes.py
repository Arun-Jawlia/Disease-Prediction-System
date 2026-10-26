from pydantic import BaseModel, Field


class DiabetesPredictionRequest(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=0, le=300)
    blood_pressure: float = Field(..., ge=0, le=200)
    skin_thickness: float = Field(..., ge=0, le=100)
    insulin: float = Field(..., ge=0, le=1000)
    bmi: float = Field(..., ge=0, le=80)
    diabetes_pedigree_function: float = Field(..., ge=0, le=3)
    age: int = Field(..., ge=1, le=120)