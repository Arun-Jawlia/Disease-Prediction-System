from app.services.container import model_service


def test_all_models_loaded():

    model_service.load_all_models()

    assert "diabetes" in model_service.models
    assert "heart" in model_service.models
    assert "liver" in model_service.models

    assert model_service.models["diabetes"] is not None
    assert model_service.models["heart"] is not None
    assert model_service.models["liver"] is not None