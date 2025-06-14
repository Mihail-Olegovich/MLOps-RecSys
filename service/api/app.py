"""FastAPI application for MLOps RecSys API."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_model_config
from .services import RecommendationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Startup
    logging.info("Starting up MLOps RecSys API")

    # Get model configuration
    model_config = get_model_config(app.state.model_type)
    if model_config is None:
        logging.error(f"Invalid model type: {app.state.model_type}")
        raise ValueError(f"Invalid model type: {app.state.model_type}")

    # Initialize recommendation service
    app.state.recommendation_service = RecommendationService(model_config)

    # Load default model if specified
    default_model = model_config.default_model
    if default_model:
        try:
            app.state.recommendation_service.load_model(default_model)
            logging.info(f"Loaded default model: {default_model}")
        except Exception as e:
            print(f"Failed to load default model {default_model}")
            logging.error(f"Failed to load default model {default_model}: {e}")
            print(f"Error loading default model: {e}")

    yield

    # Shutdown
    logging.info("Shutting down MLOps RecSys API")


def create_app(model_type: str = "als") -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="MLOps RecSys API",
        description="Recommendation System API for MLOps course",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store model type in app state
    app.state.model_type = model_type

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from .views import router

    app.include_router(router)

    return app


# Get model type from environment
model_type = os.getenv("MODEL_TYPE", "als")
app = create_app(model_type)
