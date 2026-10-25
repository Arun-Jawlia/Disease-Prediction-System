from app.core.config import settings
from app.services.model_service import (
    ModelService,
)


def main():

    service = ModelService(
        settings.model_dir
    )

    for disease in [
        "diabetes",
        "heart",
        "liver",
    ]:

        path = service.get_model_path(
            disease
        )

        print(
            f"{disease}:"
        )

        print(
            f"  {path}"
        )

        print(
            f"  Exists: "
            f"{path.exists()}"
        )


if __name__ == "__main__":
    main()