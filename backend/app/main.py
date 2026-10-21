from fastapi import FastAPI

app = FastAPI(
    title="MediPredict AI",
    description=(
        "Machine Learning-Based Disease Risk Prediction API "
        "for Diabetes, Lung Disease, and Kidney Disease."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Welcome to MediPredict AI",
        "version": "1.0.0",
        "diseases": ["Diabetes", "Lung Disease", "Kidney Disease"],
        "status": "running",
    }


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy"}