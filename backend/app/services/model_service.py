from pathlib import Path


class ModelService:

    SUPPORTED_DISEASES = {
        "diabetes",
        "heart",
        "liver",
    }

    def __init__(
        self,
        model_directory: str,
    ):

        self.model_directory = (
            Path(model_directory)
        )

    def get_model_path(
        self,
        disease: str,
    ):

        if disease not in (
            self.SUPPORTED_DISEASES
        ):

            raise ValueError(
                f"Unsupported disease: "
                f"{disease}"
            )

        return (
            self.model_directory
            / disease
            / "model.joblib"
        )

    def model_exists(
        self,
        disease: str,
    ):

        model_path = (
            self.get_model_path(
                disease
            )
        )

        return model_path.exists()