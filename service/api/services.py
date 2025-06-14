"""Services for data preprocessing, model management, and recommendations."""

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from rectools.dataset import Dataset

from service.api.config import ModelConfig, get_data_path, get_models_path

# Import metrics
from .metrics import MODEL_EVALUATION_COUNT, MODEL_RECALL_AT_K


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

    def get_current_model(self) -> Any | None:  # noqa: ANN401
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


class ModelTrainingService:
    """Service for training and saving models."""

    def __init__(self) -> None:
        """Initialize model training service."""
        self.data_path = get_data_path()
        self.models_path = get_models_path()

    def train_model(
        self,
        model_type: str,
        model_name: str,
        hyperparameters: dict[str, Any] | None = None,
        use_features: bool = False,
    ) -> dict[str, Any]:
        """
        Train a model with specified parameters.

        Args:
            model_type: Type of model ('als' or 'lightfm')
            model_name: Name for the saved model
            hyperparameters: Model hyperparameters
            use_features: Whether to use item features

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        # Validate model type
        if model_type not in ["als", "lightfm"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load training data
        train_file = self.data_path / "train.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")

        df_train = pd.read_csv(train_file)

        # Load features if needed
        df_features = None
        if use_features:
            features_file = self.data_path / "cat_features.csv"
            if not features_file.exists():
                raise FileNotFoundError(f"Features file not found: {features_file}")
            df_features = pd.read_csv(features_file)
            df_features = df_features.rename(columns={"node": "item_id"})

        # Set default hyperparameters
        if hyperparameters is None:
            hyperparameters = {}

        # Train model based on type
        if model_type == "als":
            model, final_hyperparams = self._train_als_model(
                df_train, df_features, hyperparameters, use_features
            )
        else:  # lightfm
            model, final_hyperparams = self._train_lightfm_model(
                df_train, df_features, hyperparameters, use_features
            )

        # Save model
        self.models_path.mkdir(exist_ok=True)
        model_path = self.models_path / f"{model_name}.pkl"
        model.save(str(model_path))

        training_time = time.time() - start_time

        # Prepare data statistics
        data_stats = {
            "train_rows": len(df_train),
            "unique_users": df_train["user_id"].nunique(),
            "unique_items": df_train["item_id"].nunique(),
        }

        if df_features is not None:
            data_stats["unique_features"] = df_features["feature"].nunique()

        return {
            "status": "success",
            "message": f"Model {model_name} trained successfully",
            "model_name": model_name,
            "model_path": str(model_path),
            "training_time_seconds": training_time,
            "data_stats": data_stats,
            "hyperparameters": final_hyperparams,
        }

    def _train_als_model(
        self,
        df_train: pd.DataFrame,
        df_features: pd.DataFrame | None,
        hyperparameters: dict[str, Any],
        use_features: bool,
    ) -> tuple[Any, dict[str, Any]]:
        """Train ALS model."""
        from implicit.als import AlternatingLeastSquares
        from rectools.models import ImplicitALSWrapperModel

        # Default hyperparameters for ALS
        default_params = {
            "factors": 128,
            "regularization": 1.0,
            "alpha": 1.0,
            "iterations": 10,
            "random_state": 32,
            "fit_features_together": True,
        }

        # Merge with provided hyperparameters
        final_params = {**default_params, **hyperparameters}

        # Create ALS model
        als_model = AlternatingLeastSquares(
            factors=final_params["factors"],
            regularization=final_params["regularization"],
            alpha=final_params["alpha"],
            random_state=final_params["random_state"],
            use_gpu=False,
            iterations=final_params["iterations"],
        )

        model = ImplicitALSWrapperModel(
            als_model,
            fit_features_together=final_params["fit_features_together"],
        )

        # Create dataset
        if use_features and df_features is not None:
            dataset = Dataset.construct(
                interactions_df=df_train,
                item_features_df=df_features,
                cat_item_features=["category"],
            )
        else:
            dataset = Dataset.construct(interactions_df=df_train)

        # Train model
        model.fit(dataset)

        return model, final_params

    def _train_lightfm_model(
        self,
        df_train: pd.DataFrame,
        df_features: pd.DataFrame | None,
        hyperparameters: dict[str, Any],
        use_features: bool,
    ) -> tuple[Any, dict[str, Any]]:
        """Train LightFM model."""
        from lightfm import LightFM
        from rectools.models import LightFMWrapperModel

        # Default hyperparameters for LightFM
        default_params = {
            "no_components": 128,
            "learning_rate": 0.05,
            "loss": "warp",
            "random_state": 32,
            "max_sampled": 10,
        }

        # Merge with provided hyperparameters
        final_params = {**default_params, **hyperparameters}

        # Create LightFM model
        lightfm_model = LightFM(
            no_components=final_params["no_components"],
            learning_rate=final_params["learning_rate"],
            loss=final_params["loss"],
            random_state=final_params["random_state"],
            max_sampled=final_params["max_sampled"],
        )

        model = LightFMWrapperModel(lightfm_model)

        # Create dataset
        if use_features and df_features is not None:
            dataset = Dataset.construct(
                interactions_df=df_train,
                item_features_df=df_features,
                cat_item_features=["category"],
            )
        else:
            dataset = Dataset.construct(interactions_df=df_train)

        # Train model
        model.fit(dataset)

        return model, final_params


class ModelEvaluationService:
    """Service for evaluating models on validation set."""

    def __init__(self) -> None:
        """Initialize model evaluation service."""
        self.data_path = get_data_path()

    def evaluate_model(
        self, model_name: str, model_service: ModelService
    ) -> dict[str, Any]:
        """
        Evaluate a model on validation set using Recall@40.

        Args:
            model_name: Name of the model to evaluate
            model_service: Model service instance

        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()

        # Load evaluation data
        eval_file = self.data_path / "eval.csv"
        train_file = self.data_path / "train.csv"

        if not eval_file.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_file}")
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")

        df_eval = pd.read_csv(eval_file)
        df_train = pd.read_csv(train_file)

        # Ensure model is loaded
        if not model_service.set_current_model(model_name):
            raise RuntimeError(f"Failed to load model {model_name}")

        model = model_service.get_current_model()
        if model is None:
            raise RuntimeError(f"Model {model_name} is not loaded")

        # Create dataset (same as in training)
        dataset = Dataset.construct(interactions_df=df_train)

        # Get unique evaluation users
        eval_users = df_eval["user_id"].unique()

        # Generate recommendations
        k = 40
        recos = model.recommend(
            users=eval_users,
            dataset=dataset,
            k=k,
            filter_viewed=True,
        )

        # Calculate Recall@40
        recall_at_k = self._calculate_recall_at_k(recos, df_eval, k)

        # Record metrics
        MODEL_RECALL_AT_K.labels(model_name=model_name, k=str(k)).observe(recall_at_k)
        MODEL_EVALUATION_COUNT.labels(model_name=model_name, status="success").inc()

        evaluation_time = time.time() - start_time

        return {
            "status": "success",
            "model_name": model_name,
            "recall_at_40": recall_at_k,
            "evaluation_time_seconds": evaluation_time,
            "eval_users_count": len(eval_users),
            "recommendations_generated": len(recos),
        }

    def _calculate_recall_at_k(
        self, recos: pd.DataFrame, df_eval: pd.DataFrame, k: int
    ) -> float:
        """Calculate Recall@K metric."""
        # Group recommendations by user
        user_recos = recos.groupby("user_id")["item_id"].apply(set).to_dict()

        # Group ground truth by user
        user_truth = df_eval.groupby("user_id")["item_id"].apply(set).to_dict()

        recalls = []
        for user_id in user_truth.keys():
            if user_id in user_recos:
                recommended_items = user_recos[user_id]
                true_items = user_truth[user_id]

                # Calculate recall for this user
                intersection = len(recommended_items.intersection(true_items))
                recall = intersection / len(true_items) if len(true_items) > 0 else 0.0
                recalls.append(recall)
            else:
                recalls.append(0.0)

        # Return average recall
        return sum(recalls) / len(recalls) if recalls else 0.0
