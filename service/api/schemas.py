"""Pydantic schemas for API request and response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DataPrepRequest(BaseModel):
    """Request model for data preparation."""

    eval_days_threshold: int = Field(
        default=14, description="Days threshold for evaluation split"
    )
    force_rebuild: bool = Field(
        default=False, description="Force rebuild of existing datasets"
    )


class DataPrepResponse(BaseModel):
    """Response model for data preparation."""

    status: str = Field(description="Processing status")
    message: str = Field(description="Status message")
    train_rows: int = Field(description="Number of training rows")
    eval_rows: int = Field(description="Number of evaluation rows")
    unique_users: int = Field(description="Number of unique users")
    unique_items: int = Field(description="Number of unique items")
    eval_days_threshold: int = Field(description="Days threshold used")
    processing_time_seconds: float = Field(description="Processing time in seconds")


class RecommendationItem(BaseModel):
    """Single recommendation item."""

    item_id: int = Field(description="Item ID")
    score: float = Field(description="Recommendation score")
    rank: int = Field(description="Rank in recommendation list")


class RecommendationRequest(BaseModel):
    """Request model for user recommendations."""

    user_id: int = Field(description="User ID to get recommendations for")
    top_k: int = Field(default=10, description="Number of recommendations to return")
    exclude_seen: bool = Field(
        default=True, description="Whether to exclude previously seen items"
    )


class RecommendationResponse(BaseModel):
    """Response model for user recommendations."""

    user_id: int = Field(description="User ID")
    recommendations: list[dict[str, Any]] = Field(description="List of recommendations")
    total_items: int = Field(description="Total number of recommended items")
    model_name: str = Field(description="Name of the model used")
    processing_time_seconds: float = Field(description="Processing time in seconds")


class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations."""

    user_ids: list[int] = Field(description="List of user IDs")
    top_k: int = Field(default=10, description="Number of recommendations per user")
    exclude_seen: bool = Field(
        default=True, description="Whether to exclude previously seen items"
    )


class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations."""

    recommendations: dict[str, list[dict[str, Any]]] = Field(
        description="Dictionary mapping user IDs to their recommendations"
    )
    total_users: int = Field(description="Total number of users processed")
    model_name: str = Field(description="Name of the model used")
    processing_time_seconds: float = Field(description="Processing time in seconds")


class ModelInfo(BaseModel):
    """Information about a model."""

    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    type: str = Field(description="Model type (e.g., ALS, LightFM)")
    is_loaded: bool = Field(description="Whether the model is currently loaded")
    created_at: datetime | None = Field(
        default=None, description="Model creation timestamp"
    )
    metrics: dict[str, Any] | None = Field(
        default=None, description="Model performance metrics"
    )


class ModelsListResponse(BaseModel):
    """Response model for listing models."""

    models: list[ModelInfo] = Field(description="List of available models")
    current_model: str | None = Field(
        default=None, description="Currently active model name"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(description="Error message")
    error_code: str | None = Field(default=None, description="Error code")
    timestamp: datetime | None = Field(default=None, description="Error timestamp")
