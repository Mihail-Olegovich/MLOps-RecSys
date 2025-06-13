"""API endpoints for MLOps RecSys service."""

from typing import Any

from fastapi import APIRouter, HTTPException

from service.api.schemas import (
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    DataPrepRequest,
    DataPrepResponse,
    ModelInfo,
    ModelsListResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from service.api.services import (
    DataPreprocessingService,
    ModelService,
    RecommendationService,
)

router = APIRouter()

# Global service instances
_data_service: DataPreprocessingService | None = None
_model_service: ModelService | None = None
_recommendation_service: RecommendationService | None = None


def get_data_service() -> DataPreprocessingService:
    """Get or create data preprocessing service instance."""
    global _data_service
    if _data_service is None:
        _data_service = DataPreprocessingService()
    return _data_service


def get_model_service() -> ModelService:
    """Get or create model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_recommendation_service() -> RecommendationService:
    """Get or create recommendation service instance."""
    global _recommendation_service
    if _recommendation_service is None:
        from .config import get_model_config

        model_config = get_model_config("als")
        # Pass the shared model service instance
        _recommendation_service = RecommendationService(
            model_config, get_model_service()  # type: ignore
        )
    return _recommendation_service


@router.post("/data/prepare", response_model=DataPrepResponse)
async def prepare_data(
    request: DataPrepRequest,
    service=None,  # noqa: ANN001
) -> DataPrepResponse:
    """
    Prepare training and evaluation datasets.

    This endpoint integrates the data preprocessing logic from
    mloprec/scripts/data_prep.py. It processes raw clickstream and event
    data to create clean train/eval datasets.
    """
    if service is None:
        service = get_data_service()

    try:
        result = service.prepare_data(
            eval_days_threshold=request.eval_days_threshold,
            force_rebuild=request.force_rebuild,
        )

        return DataPrepResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Data preparation failed: {str(e)}"
        ) from e


@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    service=None,  # noqa: ANN001
) -> ModelsListResponse:
    """List all available models and their status."""
    if service is None:
        service = get_model_service()

    try:
        available_models = service.list_available_models()
        loaded_models = service.list_loaded_models()

        models_info = []
        for model_name in available_models:
            models_info.append(
                ModelInfo(
                    name=model_name,
                    version="1.0",  # Could be extracted from model metadata
                    type="ALS" if "als" in model_name.lower() else "LightFM",
                    is_loaded=model_name in loaded_models,
                    created_at=None,  # Could be extracted from file metadata
                    metrics=None,  # Could be loaded from model metadata
                )
            )

        return ModelsListResponse(
            models=models_info,
            current_model=service.get_current_model_name(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(e)}"
        ) from e


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    service=None,  # noqa: ANN001
) -> dict[str, str]:
    """Load a specific model."""
    if service is None:
        service = get_model_service()

    try:
        success = service.load_model(model_name)
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} loaded successfully",
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model {model_name}"
            )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/{model_name}/activate")
async def activate_model(
    model_name: str,
    service=None,  # noqa: ANN001
) -> dict[str, str]:
    """Set a model as the current active model."""
    if service is None:
        service = get_model_service()

    try:
        success = service.set_current_model(model_name)
        if success:
            return {"status": "success", "message": f"Model {model_name} is now active"}
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to activate model {model_name}"
            )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/recommend/user", response_model=RecommendationResponse)
async def get_user_recommendations(
    request: RecommendationRequest,
    service=None,  # noqa: ANN001
) -> RecommendationResponse:
    """Get recommendations for a single user."""
    if service is None:
        service = get_recommendation_service()

    try:
        recommendations, model_name, processing_time = service.get_recommendations(
            user_id=request.user_id,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            total_items=len(recommendations),
            model_name=model_name,
            processing_time_seconds=processing_time,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {str(e)}"
        ) from e


@router.post("/recommend/batch", response_model=BatchRecommendationResponse)
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    service=None,  # noqa: ANN001
) -> BatchRecommendationResponse:
    """Get recommendations for multiple users."""
    if service is None:
        service = get_recommendation_service()

    try:
        (
            recommendations,
            model_name,
            processing_time,
        ) = service.get_batch_recommendations(
            user_ids=request.user_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )

        return BatchRecommendationResponse(
            recommendations=recommendations,
            total_users=len(request.user_ids),
            model_name=model_name,
            processing_time_seconds=processing_time,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch recommendation generation failed: {str(e)}"
        ) from e


@router.get("/stats")
async def get_api_stats(
    data_service=None,  # noqa: ANN001
    model_service=None,  # noqa: ANN001
) -> dict[str, Any]:
    """Get API statistics and status information."""
    if data_service is None:
        data_service = get_data_service()
    if model_service is None:
        model_service = get_model_service()

    try:
        from pathlib import Path

        import pandas as pd

        stats: dict[str, Any] = {
            "api_status": "healthy",
            "loaded_models": model_service.list_loaded_models(),
            "current_model": model_service.get_current_model_name(),
            "available_models": model_service.list_available_models(),
        }

        # Add data statistics if available
        train_path = Path("data/train.csv")
        if train_path.exists():
            try:
                df_train = pd.read_csv(train_path)
                stats["data_stats"] = {
                    "train_rows": len(df_train),
                    "unique_users": df_train["user_id"].nunique(),
                    "unique_items": df_train["item_id"].nunique(),
                }
            except Exception:
                stats["data_stats"] = {"error": "Unable to load training data stats"}
        else:
            stats["data_stats"] = {"error": "Training data not found"}

        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get API stats: {str(e)}"
        ) from e


@router.get("/validate/user/{user_id}")
async def validate_user(
    user_id: int,
    service=None,  # noqa: ANN001
) -> dict[str, Any]:
    """Validate if a user exists in the training data."""
    if service is None:
        service = get_recommendation_service()

    try:
        from pathlib import Path

        import pandas as pd

        train_path = Path("data/train.csv")
        if not train_path.exists():
            raise HTTPException(status_code=404, detail="Training data not found")

        df_train = pd.read_csv(train_path)
        user_exists = user_id in df_train["user_id"].values

        result = {
            "user_id": user_id,
            "exists": user_exists,
        }

        if user_exists:
            user_data = df_train[df_train["user_id"] == user_id]
            result.update(
                {
                    "interaction_count": len(user_data),
                    "unique_items": user_data["item_id"].nunique(),
                }
            )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"User validation failed: {str(e)}"
        ) from e


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "MLOps RecSys API"}
