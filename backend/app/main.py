import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import (
    api_router,
)

from app.core.config import (
    settings,
)

from app.core.logging import (
    configure_logging,
)


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

configure_logging()

logger = logging.getLogger(
    __name__
)


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

app = FastAPI(

    title=settings.app_name,

    version=settings.app_version,

    description=(
        "Machine Learning based "
        "disease risk prediction API."
    ),

    docs_url="/docs",

    redoc_url="/redoc",
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=(
        settings.cors_origin_list
    ),

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ------------------------------------------------------------
# Routers
# ------------------------------------------------------------

app.include_router(
    api_router
)


# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

@app.get(
    "/",
    tags=["Root"],
)
def root():

    return {

        "message": (
            "Disease Prediction "
            "System API"
        ),

        "version": (
            settings.app_version
        ),

        "status": "running",

    }


# ------------------------------------------------------------
# Startup
# ------------------------------------------------------------

@app.on_event("startup")
def startup_event():

    logger.info(
        "Disease Prediction API started"
    )

    logger.info(
        "Environment: %s",
        settings.app_environment,
    )

    logger.info(
        "Model directory: %s",
        settings.model_dir,
    )