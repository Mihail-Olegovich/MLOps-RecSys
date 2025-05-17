"""Evaluate and compare ALS and LightFM models."""

import os
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from rectools.dataset import Dataset
from rectools.metrics import Recall, calc_metrics
from rectools.models import ImplicitALSWrapperModel, LightFMWrapperModel

K_RECOS = 40


def load_model(model_path: str) -> ImplicitALSWrapperModel | LightFMWrapperModel:
    """Load model from pickle file."""
    if "als" in model_path:
        return ImplicitALSWrapperModel.load(model_path)
    elif "lightfm" in model_path:
        return LightFMWrapperModel.load(model_path)
    else:
        raise ValueError(f"Unknown model type in path: {model_path}")


def evaluate_model(
    model: ImplicitALSWrapperModel | LightFMWrapperModel,
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


def pull_dvc_files(file_list: list[str]) -> None:
    """Pull files from DVC storage."""
    for file_path in file_list:
        dvc_file = f"{file_path}.dvc"
        if os.path.exists(dvc_file):
            if not os.path.exists(file_path):
                print(f"File {file_path} is missing, pulling from DVC...")
                try:
                    subprocess.run(["dvc", "checkout", dvc_file], check=True)
                    print(f"Successfully pulled {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error pulling {file_path}: {e}")
            else:
                print(f"File {file_path} already exists, skipping.")
        else:
            print(f"DVC file {dvc_file} not found.")


def main() -> None:
    """Evaluate and compare different recommendation models."""
    data_files = ["data/train.csv", "data/eval.csv", "data/cat_features.csv"]

    model_files = ["models/als_model.pkl", "models/lightfm_model.pkl"]

    pull_dvc_files(data_files)

    pull_dvc_files(model_files)

    df_train = pd.read_csv("data/train.csv")
    df_eval = pd.read_csv("data/eval.csv")
    df_features = pd.read_csv("data/cat_features.csv")
    df_features.rename(columns={"node": "item_id"}, inplace=True)

    df_eval.rename(columns={"cookie": "user_id", "node": "item_id"}, inplace=True)

    catalog = df_train["item_id"].unique().tolist()

    models = {
        "ALS": load_model("models/als_model.pkl"),
        "LightFM": load_model("models/lightfm_model.pkl"),
    }

    results = {}
    for model_name, model in models.items():
        if model_name == "ALS":
            dataset = Dataset.construct(
                interactions_df=df_train,
                item_features_df=df_features,
                cat_item_features=["category"],
            )
        else:
            dataset = Dataset.construct(
                interactions_df=df_train,
            )
        print(f"Evaluating {model_name}...")
        results[model_name] = evaluate_model(model, dataset, df_eval, catalog)

    print("\nResults:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    with open("models/evaluation_results.txt", "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("======================\n\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
