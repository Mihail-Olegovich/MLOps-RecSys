"""API endpoints for MLOps RecSys service."""

from typing import Any

from fastapi import APIRouter, HTTPException, Response

from service.api.metrics import (
    DATA_PROCESSING_COUNT,
    MODEL_EVALUATION_COUNT,
    MODEL_LOAD_COUNT,
    MODEL_TRAINING_COUNT,
    RECOMMENDATION_COUNT,
    REQUEST_COUNT,
)
from service.api.schemas import (
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    DataPrepRequest,
    DataPrepResponse,
    EvaluateModelRequest,
    EvaluateModelResponse,
    ModelInfo,
    ModelsListResponse,
    RecommendationRequest,
    RecommendationResponse,
    TrainModelRequest,
    TrainModelResponse,
)
from service.api.services import (
    DataPreprocessingService,
    ModelEvaluationService,
    ModelService,
    ModelTrainingService,
    RecommendationService,
)

router = APIRouter()

# Global service instances
_data_service: DataPreprocessingService | None = None
_model_service: ModelService | None = None
_recommendation_service: RecommendationService | None = None
_training_service: ModelTrainingService | None = None
_evaluation_service: ModelEvaluationService | None = None


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


def get_training_service() -> ModelTrainingService:
    """Get or create model training service instance."""
    global _training_service
    if _training_service is None:
        _training_service = ModelTrainingService()
    return _training_service


def get_evaluation_service() -> ModelEvaluationService:
    """Get or create model evaluation service instance."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = ModelEvaluationService()
    return _evaluation_service


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
        REQUEST_COUNT.labels(
            method="POST", endpoint="/data/prepare", status_code="200"
        ).inc()
        DATA_PROCESSING_COUNT.labels(status="success").inc()

        result = service.prepare_data(
            eval_days_threshold=request.eval_days_threshold,
            force_rebuild=request.force_rebuild,
        )

        return DataPrepResponse(**result)

    except FileNotFoundError as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/data/prepare", status_code="404"
        ).inc()
        DATA_PROCESSING_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/data/prepare", status_code="500"
        ).inc()
        DATA_PROCESSING_COUNT.labels(status="error").inc()
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
        REQUEST_COUNT.labels(method="GET", endpoint="/models", status_code="200").inc()
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
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/load", status_code="200"
        ).inc()
        success = service.load_model(model_name)
        if success:
            MODEL_LOAD_COUNT.labels(model_name=model_name, status="success").inc()
            return {
                "status": "success",
                "message": f"Model {model_name} loaded successfully",
            }
        else:
            MODEL_LOAD_COUNT.labels(model_name=model_name, status="error").inc()
            REQUEST_COUNT.labels(
                method="POST", endpoint="/models/load", status_code="500"
            ).inc()
            raise HTTPException(
                status_code=500, detail=f"Failed to load model {model_name}"
            )

    except FileNotFoundError as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/load", status_code="404"
        ).inc()
        MODEL_LOAD_COUNT.labels(model_name=model_name, status="error").inc()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/load", status_code="500"
        ).inc()
        MODEL_LOAD_COUNT.labels(model_name=model_name, status="error").inc()
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
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/activate", status_code="200"
        ).inc()
        success = service.set_current_model(model_name)
        if success:
            return {"status": "success", "message": f"Model {model_name} is now active"}
        else:
            REQUEST_COUNT.labels(
                method="POST", endpoint="/models/activate", status_code="500"
            ).inc()
            raise HTTPException(
                status_code=500, detail=f"Failed to activate model {model_name}"
            )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/train", response_model=TrainModelResponse)
async def train_model(
    request: TrainModelRequest,
    service=None,  # noqa: ANN001
) -> TrainModelResponse:
    """Train a new model with specified parameters."""
    if service is None:
        service = get_training_service()

    try:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/train", status_code="200"
        ).inc()
        MODEL_TRAINING_COUNT.labels(
            model_type=request.model_type, status="success"
        ).inc()

        result = service.train_model(
            model_type=request.model_type,
            model_name=request.model_name,
            hyperparameters=request.hyperparameters,
            use_features=request.use_features,
        )

        return TrainModelResponse(**result)

    except ValueError as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/train", status_code="400"
        ).inc()
        MODEL_TRAINING_COUNT.labels(model_type=request.model_type, status="error").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/train", status_code="404"
        ).inc()
        MODEL_TRAINING_COUNT.labels(model_type=request.model_type, status="error").inc()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/train", status_code="500"
        ).inc()
        MODEL_TRAINING_COUNT.labels(model_type=request.model_type, status="error").inc()
        raise HTTPException(
            status_code=500, detail=f"Model training failed: {str(e)}"
        ) from e


@router.post("/models/evaluate", response_model=EvaluateModelResponse)
async def evaluate_model(
    request: EvaluateModelRequest,
    evaluation_service=None,  # noqa: ANN001
    model_service=None,  # noqa: ANN001
) -> EvaluateModelResponse:
    """Evaluate a model on validation set using Recall@40."""
    if evaluation_service is None:
        evaluation_service = get_evaluation_service()
    if model_service is None:
        model_service = get_model_service()

    try:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/evaluate", status_code="200"
        ).inc()

        result = evaluation_service.evaluate_model(
            model_name=request.model_name,
            model_service=model_service,
        )

        MODEL_EVALUATION_COUNT.labels(
            model_name=request.model_name, status="success"
        ).inc()

        return EvaluateModelResponse(**result)

    except FileNotFoundError as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/evaluate", status_code="404"
        ).inc()
        MODEL_EVALUATION_COUNT.labels(
            model_name=request.model_name, status="error"
        ).inc()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/evaluate", status_code="400"
        ).inc()
        MODEL_EVALUATION_COUNT.labels(
            model_name=request.model_name, status="error"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/models/evaluate", status_code="500"
        ).inc()
        MODEL_EVALUATION_COUNT.labels(
            model_name=request.model_name, status="error"
        ).inc()
        raise HTTPException(
            status_code=500, detail=f"Model evaluation failed: {str(e)}"
        ) from e


@router.post("/recommend/user", response_model=RecommendationResponse)
async def get_user_recommendations(
    request: RecommendationRequest,
    service=None,  # noqa: ANN001
) -> RecommendationResponse:
    """Get recommendations for a single user."""
    if service is None:
        service = get_recommendation_service()

    try:
        REQUEST_COUNT.labels(
            method="POST", endpoint="/recommend/user", status_code="200"
        ).inc()

        recommendations, model_name, processing_time = service.get_recommendations(
            user_id=request.user_id,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )

        RECOMMENDATION_COUNT.labels(
            model_name=model_name or "unknown", request_type="single"
        ).inc()

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
        REQUEST_COUNT.labels(
            method="POST", endpoint="/recommend/batch", status_code="200"
        ).inc()

        (
            recommendations,
            model_name,
            processing_time,
        ) = service.get_batch_recommendations(
            user_ids=request.user_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )

        RECOMMENDATION_COUNT.labels(
            model_name=model_name or "unknown", request_type="batch"
        ).inc()

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
        REQUEST_COUNT.labels(method="GET", endpoint="/stats", status_code="200").inc()

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
        REQUEST_COUNT.labels(
            method="GET", endpoint="/validate/user", status_code="200"
        ).inc()

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
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code="200").inc()
    return {"status": "healthy"}


@router.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    from .metrics import get_metrics

    metrics = get_metrics()
    return Response(content=metrics, media_type="text/plain")


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    REQUEST_COUNT.labels(method="GET", endpoint="/", status_code="200").inc()
    return {"message": "MLOps RecSys API"}
