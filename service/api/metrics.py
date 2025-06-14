"""Prometheus metrics for MLOps RecSys API."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from prometheus_client import Counter, Histogram, Info, generate_latest  # type: ignore

# API request metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)

# Model training metrics
MODEL_TRAINING_COUNT = Counter(
    "model_training_total",
    "Total number of model training requests",
    ["model_type", "status"],
)

MODEL_TRAINING_DURATION = Histogram(
    "model_training_duration_seconds",
    "Model training duration in seconds",
    ["model_type"],
)

# Recommendation metrics
RECOMMENDATION_COUNT = Counter(
    "recommendations_total",
    "Total number of recommendation requests",
    ["model_name", "request_type"],
)

RECOMMENDATION_DURATION = Histogram(
    "recommendation_duration_seconds",
    "Recommendation generation duration in seconds",
    ["model_name", "request_type"],
)

# Data processing metrics
DATA_PROCESSING_COUNT = Counter(
    "data_processing_total",
    "Total number of data processing requests",
    ["status"],
)

DATA_PROCESSING_DURATION = Histogram(
    "data_processing_duration_seconds",
    "Data processing duration in seconds",
)

# Model management metrics
MODEL_LOAD_COUNT = Counter(
    "model_loads_total",
    "Total number of model load requests",
    ["model_name", "status"],
)

# Model evaluation metrics
MODEL_RECALL_AT_K = Histogram(
    "model_recall_at_k",
    "Model Recall@K on validation set",
    ["model_name", "k"],
    buckets=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
)

MODEL_EVALUATION_COUNT = Counter(
    "model_evaluations_total",
    "Total number of model evaluations",
    ["model_name", "status"],
)

# Application info
APP_INFO = Info("app_info", "Application information")
APP_INFO.info(
    {
        "version": "1.0.0",
        "name": "MLOps RecSys API",
        "description": "Recommendation System API for MLOps course",
    }
)


def track_request_metrics(
    endpoint: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to track request metrics."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> object:
            start_time = time.time()
            status_code = "200"
            method = "POST"  # Most of our endpoints are POST

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = "500"
                if hasattr(e, "status_code"):
                    status_code = str(e.status_code)
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).inc()
                REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
                    duration
                )

        return wrapper

    return decorator


def track_training_metrics(
    model_type: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to track model training metrics."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> object:
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                MODEL_TRAINING_COUNT.labels(model_type=model_type, status=status).inc()
                MODEL_TRAINING_DURATION.labels(model_type=model_type).observe(duration)

        return wrapper

    return decorator


def track_recommendation_metrics(
    model_name: str, request_type: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to track recommendation metrics."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> object:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                RECOMMENDATION_COUNT.labels(
                    model_name=model_name, request_type=request_type
                ).inc()
                RECOMMENDATION_DURATION.labels(
                    model_name=model_name, request_type=request_type
                ).observe(duration)

        return wrapper

    return decorator


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    return str(generate_latest().decode("utf-8"))
