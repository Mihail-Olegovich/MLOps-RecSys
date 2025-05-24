"""Train LightFM model."""

import pandas as pd
from lightfm import LightFM
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel

from mloprec.tracking import init_task, log_model, log_parameters

K_RECOS = 40
RANDOM_STATE = 32
ITERATIONS = 10
FACTOR = 60
REGULARIZATION = 0.1
ALPHA = 1.0


def main() -> None:
    """Main function to train the ALS model."""
    # Initialize ClearML task
    task = init_task(task_name="LightFM Model without Features")

    # Log hyperparameters
    hyperparams = {
        "k_recos": K_RECOS,
        "random_state": RANDOM_STATE,
        "iterations": ITERATIONS,
        "factors": FACTOR,
        "regularization": REGULARIZATION,
        "alpha": ALPHA,
        "loss": "bpr",
    }
    log_parameters(task, hyperparams)

    # Pull data using DVC
    import subprocess

    subprocess.run(["dvc", "pull", "data/train.csv.dvc"], check=True)

    df_train = pd.read_csv("data/train.csv")
    df_features = pd.read_csv("data/cat_features.csv")
    df_features.rename(columns={"node": "item_id"}, inplace=True)

    # Log data statistics
    data_stats = {
        "train_rows": len(df_train),
        "unique_users": df_train["user_id"].nunique(),
        "unique_items": df_train["item_id"].nunique(),
    }
    log_parameters(task, {"data_stats": data_stats})

    model = LightFMWrapperModel(
        LightFM(
            no_components=FACTOR,
            loss="bpr",
        ),
        epochs=ITERATIONS,
    )

    dataset = Dataset.construct(
        interactions_df=df_train,
    )

    model.fit(dataset)

    model_path = "models/lightfm_model.pkl"
    model.save(model_path)

    # Log model as an artifact
    log_model(task, model_path=model_path, model_name="LightFM Model without Features")

    # Complete the task
    task.close()


if __name__ == "__main__":
    main()
