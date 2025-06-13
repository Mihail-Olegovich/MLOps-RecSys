"""Services for data preprocessing, model management, and recommendations."""

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from rectools.dataset import Dataset

from service.api.config import ModelConfig, get_data_path, get_models_path


class DataPreprocessingService:
    """Service for data preprocessing operations."""

    def __init__(self) -> None:
        """Initialize data preprocessing service."""
        self.data_path = get_data_path()

    def prepare_data(
        self, eval_days_threshold: int = 14, force_rebuild: bool = False
    ) -> dict[str, Any]:
        """
        Prepare training and evaluation datasets.

        Integrates the logic from mloprec/scripts/data_prep.py.
        """
        from datetime import timedelta

        train_file = self.data_path / "train.csv"
        eval_file = self.data_path / "eval.csv"

        # Check if files exist and force_rebuild is False
        if not force_rebuild and train_file.exists() and eval_file.exists():
            # Load existing files for stats
            df_train = pd.read_csv(train_file)
            df_eval = pd.read_csv(eval_file)

            return {
                "status": "loaded_existing",
                "message": "Using existing preprocessed data",
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
                "unique_users": df_train["user_id"].nunique(),
                "unique_items": df_train["item_id"].nunique(),
                "eval_days_threshold": eval_days_threshold,
                "processing_time_seconds": 0,
            }

        # Load raw data
        clickstream_file = self.data_path / "fclickstream.csv"
        event_file = self.data_path / "event.csv"

        if not clickstream_file.exists():
            raise FileNotFoundError(f"Clickstream file not found: {clickstream_file}")
        if not event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file}")

        start_time = time.time()

        df_clickstream = pd.read_csv(clickstream_file)
        df_event = pd.read_csv(event_file)

        df_clickstream["event_date"] = pd.to_datetime(df_clickstream["event_date"])

        threshold = df_clickstream["event_date"].max() - timedelta(
            days=eval_days_threshold
        )

        df_train = df_clickstream[df_clickstream["event_date"] <= threshold]
        df_eval = df_clickstream[df_clickstream["event_date"] > threshold][
            ["cookie", "node", "event"]
        ]

        df_eval = df_eval.merge(
            df_train[["cookie", "node"]],
            on=["cookie", "node"],
            how="left",
            indicator=True,
        )
        df_eval = df_eval[df_eval["_merge"] == "left_only"].drop("_merge", axis=1)

        contact_events = df_event[df_event["is_contact"] == 1]["event"].unique()
        df_eval = df_eval[df_eval["event"].isin(contact_events)]

        df_eval = df_eval[df_eval["cookie"].isin(df_train["cookie"].unique())]
        df_eval = df_eval[df_eval["node"].isin(df_train["node"].unique())]

        df_eval = df_eval.drop_duplicates(["cookie", "node"])
        df_eval.rename(columns={"cookie": "user_id", "node": "item_id"}, inplace=True)

        df_train.rename(
            columns={"cookie": "user_id", "node": "item_id", "event_date": "datetime"},
            inplace=True,
        )
        df_train["weight"] = 1.0
        df_train["datetime"] = pd.to_datetime(df_train["datetime"])
        df_train = df_train[["user_id", "item_id", "datetime", "weight"]]

        # Save processed data
        self.data_path.mkdir(exist_ok=True)
        df_train.to_csv(train_file, index=False)
        df_eval.to_csv(eval_file, index=False)

        processing_time = time.time() - start_time

        return {
            "status": "success",
            "message": "Data preprocessing completed successfully",
            "train_rows": len(df_train),
            "eval_rows": len(df_eval),
            "unique_users": df_train["user_id"].nunique(),
            "unique_items": df_train["item_id"].nunique(),
            "eval_days_threshold": eval_days_threshold,
            "processing_time_seconds": processing_time,
        }


class ModelService:
    """Service for model management and loading."""

    def __init__(self) -> None:
        """Initialize model service."""
        self.models_path = get_models_path()
        self.loaded_models: dict[str, Any] = {}
        self.current_model: Any | None = None
        self.current_model_name: str | None = None

    def list_available_models(self) -> list[str]:
        """List all available model files."""
        if not self.models_path.exists():
            return []

        model_files = []
        for file_path in self.models_path.glob("*.pkl"):
            model_files.append(file_path.stem)

        return sorted(model_files)

    def list_loaded_models(self) -> list[str]:
        """List all currently loaded models."""
        return list(self.loaded_models.keys())

    def load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        model_file = self.models_path / f"{model_name}.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)

            self.loaded_models[model_name] = model
            print(f"Model {model_name} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            raise

    def set_current_model(self, model_name: str) -> bool:
        """Set a model as the current active model."""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return False

        self.current_model = self.loaded_models[model_name]
        self.current_model_name = model_name
        return True

    def get_current_model(self) -> object | None:  # noqa: ANN401
        """Get the currently loaded model object."""
        return self.current_model

    def get_current_model_name(self) -> str | None:
        """Get the name of the currently loaded model."""
        return self.current_model_name

    def _get_available_model_files(self) -> set[str]:
        """Get set of available model files."""
        if not self.models_path.exists():
            return set()

        return {f.stem for f in self.models_path.glob("*.pkl")}

    def auto_load_default_model(self) -> bool:
        """Auto-load the first available model as default."""
        available_models = self.list_available_models()

        if not available_models:
            print("No models available for auto-loading")
            return False

        # Try to load the first available model
        default_model = available_models[0]
        try:
            return self.set_current_model(default_model)
        except Exception as e:
            print(f"Failed to auto-load default model {default_model}: {e}")
            return False

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific model."""
        model_file = self.models_path / f"{model_name}.pkl"

        info = {
            "name": model_name,
            "exists": model_file.exists(),
            "loaded": model_name in self.loaded_models,
            "is_current": model_name == self.current_model_name,
        }

        if model_file.exists():
            info["file_size"] = model_file.stat().st_size
            info["file_path"] = str(model_file)

        return info


class RecommendationService:
    """Service for generating recommendations."""

    def __init__(
        self, model_config: ModelConfig, model_service: ModelService | None = None
    ) -> None:
        """Initialize recommendation service."""
        self.model_config = model_config
        self.model_service = (
            model_service if model_service is not None else ModelService()
        )
        self.dataset: Dataset | None = None
        self.dataset_cache: dict[str, Any] = {}

    def load_model(self, model_name: str) -> bool:
        """Load a model."""
        return self.model_service.load_model(model_name)

    def _load_training_data(self) -> Dataset:
        """Load training data and create Dataset."""
        data_path = get_data_path()
        train_file = data_path / "train.csv"

        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")

        df_train = pd.read_csv(train_file)

        # Create interactions dataset
        dataset = Dataset.construct(
            interactions_df=df_train,
        )

        return dataset

    def _get_dataset_for_model(self) -> Dataset:
        """Get or create dataset for current model."""
        model_name = self.model_service.get_current_model_name()

        if not model_name:
            raise RuntimeError("No model is currently loaded")

        # Use cached dataset if available
        if model_name in self.dataset_cache:
            return self.dataset_cache[model_name]

        # Load training data
        dataset = self._load_training_data()

        # Both models were trained without features
        needs_features = False

        if needs_features:
            # This would be for models that need item features
            pass
        else:
            # For models without features, use dataset as-is
            pass

        # Cache the dataset
        self.dataset_cache[model_name] = dataset
        return dataset

    def get_recommendations(
        self, user_id: int, top_k: int = 10, exclude_seen: bool = True
    ) -> tuple[list[dict[str, Any]], str, float]:
        """Get recommendations for a single user."""
        start_time = time.time()

        model = self.model_service.get_current_model()
        model_name = self.model_service.get_current_model_name()

        if model is None or model_name is None:
            raise RuntimeError("No model is currently loaded")

        dataset = self._get_dataset_for_model()

        # Check if user exists
        if user_id not in dataset.user_id_map.external_ids:
            raise RuntimeError(f"User {user_id} not found in training data")

        # Get user recommendations
        user_ids = np.array([user_id])
        recs = model.recommend(  # type: ignore
            users=user_ids,
            dataset=dataset,
            k=top_k,
            filter_viewed=exclude_seen,
        )

        # Convert to list of dicts
        recommendations = []
        for _, row in recs.iterrows():
            recommendations.append(
                {
                    "item_id": int(row["item_id"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                }
            )

        processing_time = time.time() - start_time

        return recommendations, model_name, processing_time

    def get_batch_recommendations(
        self, user_ids: list[int], top_k: int = 10, exclude_seen: bool = True
    ) -> tuple[dict[str, list[dict[str, Any]]], str, float]:
        """Get recommendations for multiple users."""
        start_time = time.time()

        model = self.model_service.get_current_model()
        model_name = self.model_service.get_current_model_name()

        if model is None or model_name is None:
            raise RuntimeError("No model is currently loaded")

        dataset = self._get_dataset_for_model()

        # Filter valid user IDs
        valid_user_ids = [
            uid for uid in user_ids if uid in dataset.user_id_map.external_ids
        ]

        if not valid_user_ids:
            raise RuntimeError("No valid users found in training data")

        # Get batch recommendations
        user_ids_array = np.array(valid_user_ids)
        recs = model.recommend(  # type: ignore
            users=user_ids_array,
            dataset=dataset,
            k=top_k,
            filter_viewed=exclude_seen,
        )

        # Group by user
        recommendations = {}
        for user_id in valid_user_ids:
            user_recs = recs[recs["user_id"] == user_id]
            recommendations[str(user_id)] = [
                {
                    "item_id": int(row["item_id"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                }
                for _, row in user_recs.iterrows()
            ]

        processing_time = time.time() - start_time

        return recommendations, model_name, processing_time
