import json
from pathlib import Path

from app.core.config import settings
from app.ml.loader import ModelLoader


class ModelService:

    SUPPORTED_DISEASES = {
        "diabetes",
        "heart",
        "liver",
    }

    def __init__(self):
        self.model_directory = Path(settings.model_dir)

        self.loader = ModelLoader(
            models_directory=self.model_directory,
            checksum_enabled=settings.model_checksum_enabled,
        )

        self.models = {}
        self.metadata = {}

    def load_all_models(self):

        for disease in self.SUPPORTED_DISEASES:

            self.models[disease] = (
                self.loader.load_model(disease)
            )

            metadata_path = (
                self.model_directory
                / disease
                / "metadata.json"
            )

            if metadata_path.exists():

                with metadata_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    self.metadata[disease] = json.load(file)

            else:
                self.metadata[disease] = {}

    def get_model(self, disease: str):

        if disease not in self.SUPPORTED_DISEASES:
            raise ValueError(
                f"Unsupported disease: {disease}"
            )

        if disease not in self.models:
            raise RuntimeError(
                f"Model not loaded for: {disease}"
            )

        return self.models[disease]

    def get_metadata(self, disease: str):

        if disease not in self.metadata:
            raise RuntimeError(
                f"Metadata not loaded for: {disease}"
            )

        return self.metadata[disease]