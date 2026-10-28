from fastapi import APIRouter, HTTPException

from app.schemas.diabetes import (
    DiabetesPredictionRequest,
)
from app.schemas.heart import (
    HeartPredictionRequest,
)
from app.schemas.liver import (
    LiverPredictionRequest,
)
from app.schemas.response import (
    PredictionResponse,
)

from app.services.container import prediction_service

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)



@router.post(
    "/diabetes",
    response_model=PredictionResponse,
)
def predict_diabetes(
    request: DiabetesPredictionRequest,
):

    try:

        features = {
            "Pregnancies": request.pregnancies,
            "Glucose": request.glucose,
            "BloodPressure": request.blood_pressure,
            "SkinThickness": request.skin_thickness,
            "Insulin": request.insulin,
            "BMI": request.bmi,
            "DiabetesPedigreeFunction": (
                request.diabetes_pedigree_function
            ),
            "Age": request.age,
        }

        return prediction_service.predict(
            "diabetes",
            features,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post(
    "/heart",
    response_model=PredictionResponse,
)
def predict_heart(
    request: HeartPredictionRequest,
):

    try:

        features = {
            "age": request.age,
            "sex": request.sex,
            "cp": request.cp,
            "trestbps": request.trestbps,
            "chol": request.chol,
            "fbs": request.fbs,
            "restecg": request.restecg,
            "thalach": request.thalach,
            "exang": request.exang,
            "oldpeak": request.oldpeak,
            "slope": request.slope,
            "ca": request.ca,
            "thal": request.thal,
        }

        return prediction_service.predict(
            "heart",
            features,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post(
    "/liver",
    response_model=PredictionResponse,
)
def predict_liver(
    request: LiverPredictionRequest,
):

    try:

        features = {
            "Age": request.age,
            "Gender": request.gender,
            "Total_Bilirubin": request.total_bilirubin,
            "Direct_Bilirubin": request.direct_bilirubin,
            "Alkaline_Phosphotase": (
                request.alkaline_phosphotase
            ),
            "Alamine_Aminotransferase": (
                request.alamine_aminotransferase
            ),
            "Aspartate_Aminotransferase": (
                request.aspartate_aminotransferase
            ),
            "Total_Protiens": request.total_proteins,
            "Albumin": request.albumin,
            "Albumin_and_Globulin_Ratio": (
                request.albumin_and_globulin_ratio
            ),
        }

        return prediction_service.predict(
            "liver",
            features,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )