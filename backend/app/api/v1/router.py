from fastapi import APIRouter

from app.api.v1.health import (
    router as health_router,
)

from app.api.v1.diseases import (
    router as diseases_router,
)


api_router = APIRouter(
    prefix="/api/v1"
)


api_router.include_router(
    health_router
)

api_router.include_router(
    diseases_router
)