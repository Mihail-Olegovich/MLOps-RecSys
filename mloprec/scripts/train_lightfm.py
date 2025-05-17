"""Train LightFM model."""

import pandas as pd
from lightfm import LightFM
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel

K_RECOS = 40
RANDOM_STATE = 32
ITERATIONS = 10
FACTOR = 60
REGULARIZATION = 0.1
ALPHA = 1.0


def main() -> None:
    """Main function to train the ALS model."""
    df_train = pd.read_csv("data/train.csv")
    df_features = pd.read_csv("data/cat_features.csv")
    df_features.rename(columns={"node": "item_id"}, inplace=True)

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

    model.save("models/lightfm_model.pkl")


if __name__ == "__main__":
    main()
