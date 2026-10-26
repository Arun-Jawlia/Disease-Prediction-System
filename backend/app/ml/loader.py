import logging
from pathlib import Path

import joblib

from app.utils.checksum import verify_sha256

logger = logging.getLogger(__name__)


class ModelLoader:

    def __init__(
        self,
        models_directory: Path,
        checksum_enabled: bool = True,
    ):
        self.models_directory = Path(models_directory)
        self.checksum_enabled = checksum_enabled

    def load_model(self, disease: str):
        model_path = (
            self.models_directory
            / disease
            / "model.joblib"
        )

        checksum_path = (
            self.models_directory
            / disease
            / "checksum.sha256"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found for disease: {disease}"
            )

        if self.checksum_enabled:
            if not verify_sha256(
                model_path,
                checksum_path,
            ):
                raise ValueError(
                    f"Checksum verification failed for: {disease}"
                )

        logger.info(
            "Loading model for %s from %s",
            disease,
            model_path,
        )

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            raise ValueError(
                f"Invalid model artifact for: {disease}"
            )

        return model