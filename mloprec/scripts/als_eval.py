"""Evaluate and compare ALS and LightFM models."""

from typing import Any

import numpy as np
import pandas as pd
from rectools.dataset import Dataset
from rectools.metrics import Recall, calc_metrics
from rectools.models import ImplicitALSWrapperModel

K_RECOS = 40


def load_model(model_path: str) -> ImplicitALSWrapperModel:
    """Load model from pickle file."""
    return ImplicitALSWrapperModel.load(model_path)


def evaluate_model(
    model: ImplicitALSWrapperModel,
    dataset: Dataset,
    df_eval: pd.DataFrame,
    catalog: list[Any] | pd.Series | np.ndarray,
) -> dict[str, float]:
    """Evaluate model and return metrics."""
    metrics_name = {
        "Recall": Recall,
    }
    metrics = {}
    for metric_name, metric in metrics_name.items():
        for k in [40]:
            metrics[f"{metric_name}@{k}"] = metric(k=k)

    eval_users = df_eval["user_id"].unique()

    recos = model.recommend(
        users=eval_users,
        dataset=dataset,
        k=K_RECOS,
        filter_viewed=True,
    )

    metric_values = calc_metrics(metrics, recos, df_eval, dataset, catalog)
    return dict(metric_values)


def main() -> None:
    """Evaluate and compare different recommendation models."""
    df_train = pd.read_csv("data/train.csv")
    df_eval = pd.read_csv("data/eval.csv")
    df_features = pd.read_csv("data/cat_features.csv")
    df_features.rename(columns={"node": "item_id"}, inplace=True)

    df_eval.rename(columns={"cookie": "user_id", "node": "item_id"}, inplace=True)

    catalog = df_train["item_id"].unique().tolist()

    model = load_model("models/als_model.pkl")

    dataset = Dataset.construct(
        interactions_df=df_train,
        item_features_df=df_features,
        cat_item_features=["category"],
    )

    results = evaluate_model(model, dataset, df_eval, catalog)

    with open("models/als_evaluation_results.txt", "a") as f:
        f.write("ALS Evaluation Results\n")
        f.write("======================\n\n")
        for metric_name, value in results.items():
            f.write(f"  {metric_name}: {value:.4f}\n")
        f.write("\n")


if __name__ == "__main__":
    main()
