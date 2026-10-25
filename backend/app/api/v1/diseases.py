from fastapi import APIRouter

router = APIRouter(
    prefix="/diseases",
    tags=["Diseases"],
)


DISEASES = [
    {
        "id": "diabetes",
        "name": "Type 2 Diabetes",
    },
    {
        "id": "heart",
        "name": "Heart Disease",
    },
    {
        "id": "liver",
        "name": "Liver Disease",
    },
]


@router.get("/")
def get_diseases():

    return {
        "count": len(DISEASES),
        "diseases": DISEASES,
    }