"""Configuration settings for the API service."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """API configuration settings."""

    # API settings
    api_title: str = "MLOps RecSys API"
    api_version: str = "1.0.0"
    api_description: str = (
        "API service for recommendation system with data preprocessing "
        "and model inference"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Enable reload")
    log_level: str = Field(default="info", description="Log level")

    # Directory paths
    data_dir: str = Field(default="data", description="Data directory")
    models_dir: str = Field(default="models", description="Models directory")

    # Model settings
    default_model: str | None = Field(default=None, description="Default model")
    auto_load_models: bool = Field(default=True, description="Auto load models")
    available_models: list[str] = Field(
        default_factory=lambda: ["als_model", "lightfm_model"],
        description="Available models",
    )

    # Recommendation settings
    max_recommendations: int = Field(default=100, description="Max recommendations")
    default_top_k: int = Field(default=10, description="Default top k")
    cache_recommendations: bool = Field(
        default=False, description="Cache recommendations"
    )

    # Data preprocessing settings
    default_eval_days_threshold: int = Field(
        default=14, description="Default eval days threshold"
    )
    max_eval_days_threshold: int = Field(
        default=365, description="Max eval days threshold"
    )

    # Performance settings
    max_batch_size: int = Field(default=1000, description="Max batch size")
    request_timeout: int = Field(default=300, description="Request timeout")

    # CORS settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS origins"
    )
    cors_credentials: bool = Field(default=True, description="CORS credentials")
    cors_methods: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS methods"
    )
    cors_headers: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS headers"
    )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"


class ModelConfig(BaseSettings):
    """Model-specific configuration."""

    model_name: str = Field(description="Model name")
    model_version: str = Field(default="1.0", description="Model version")
    model_type: str = Field(description="Model type")  # "als" or "lightfm"
    model_path: str | None = Field(default=None, description="Model path")
    default_model: str | None = Field(default=None, description="Default model to load")

    # Model parameters
    factors: int = Field(default=128, description="Number of factors")
    regularization: float = Field(default=1.0, description="Regularization parameter")
    iterations: int = Field(default=10, description="Number of iterations")
    alpha: float = Field(default=1.0, description="Alpha parameter")

    # Performance settings
    use_gpu: bool = Field(default=False, description="Use GPU")
    num_threads: int = Field(default=1, description="Number of threads")

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = APISettings()

# Model configurations for different model types
MODEL_CONFIGS = {
    "als": ModelConfig(
        model_name="als",
        model_type="als",
        factors=128,
        regularization=1.0,
        iterations=10,
        alpha=1.0,
        default_model="als_model.pkl",
    ),
    "lightfm": ModelConfig(
        model_name="lightfm",
        model_type="lightfm",
        factors=128,
        regularization=0.1,
        iterations=20,
        default_model="lightfm_model.pkl",
    ),
}


def get_model_config(model_name: str) -> ModelConfig | None:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name)


def get_data_path() -> Path:
    """Get the data directory path."""
    return Path(settings.data_dir)


def get_models_path() -> Path:
    """Get the models directory path."""
    return Path(settings.models_dir)
