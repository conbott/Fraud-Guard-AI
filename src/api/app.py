"""FastAPI main application entrypoint for FraudGuard AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router


def create_app() -> FastAPI:
    """Application factory for FraudGuard AI REST service."""
    app = FastAPI(
        title="FraudGuard AI API",
        description=(
            "Production-grade credit card fraud detection and risk estimation service. "
            "Exposes calibrated prediction probabilities, threshold decisions, and audit telemetry."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for local dashboards or frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    # Local Windows-native execution
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
