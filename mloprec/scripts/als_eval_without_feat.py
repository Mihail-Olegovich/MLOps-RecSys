"""Evaluate ALS model without features."""

from typing import Any

import numpy as np
import pandas as pd
from rectools.dataset import Dataset
from rectools.metrics import Recall, calc_metrics
from rectools.models import ImplicitALSWrapperModel

from mloprec.tracking import (
    get_model_from_clearml,
    init_task,
    log_artifact,
    log_metrics,
    log_parameters,
)

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
    # Initialize ClearML task
    task = init_task(
        task_name="ALS without Features Evaluation",
        task_type="testing",
    )

    # Pull data using DVC
    import subprocess

    subprocess.run(
        ["dvc", "pull", "data/train.csv.dvc", "data/eval.csv.dvc"], check=True
    )

    df_train = pd.read_csv("data/train.csv")
    df_eval = pd.read_csv("data/eval.csv")

    df_eval.rename(columns={"cookie": "user_id", "node": "item_id"}, inplace=True)

    # Log data statistics
    data_stats = {
        "train_rows": len(df_train),
        "eval_rows": len(df_eval),
        "unique_eval_users": df_eval["user_id"].nunique(),
        "unique_eval_items": df_eval["item_id"].nunique(),
    }
    log_parameters(task, {"data_stats": data_stats})

    catalog = df_train["item_id"].unique().tolist()

    model_path = get_model_from_clearml(
        model_name="ALS Model without Features",
        task_id="89b577a641ca42168fbb975a3d63ee66",
    )
    task.get_logger().report_text(f"Модель загружена из ClearML: {model_path}")

    model = load_model(model_path)

    dataset = Dataset.construct(
        interactions_df=df_train,
    )

    results = evaluate_model(model, dataset, df_eval, catalog)

    # Log metrics to ClearML
    for metric_name, value in results.items():
        log_metrics(task, "Evaluation Metrics", metric_name, value)

    # Save results to file
    with open("models/als_without_feat_evaluation_results.txt", "a") as f:
        f.write("ALS Evaluation Results\n")
        f.write("======================\n\n")
        for metric_name, value in results.items():
            f.write(f"  {metric_name}: {value:.4f}\n")
        f.write("\n")

    # Log results file as artifact
    log_artifact(
        task, "evaluation_results", "models/als_without_feat_evaluation_results.txt"
    )

    # Complete the task
    task.close()


if __name__ == "__main__":
    main()
