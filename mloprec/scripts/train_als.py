"""Train ALS model."""

import pandas as pd
from implicit.als import AlternatingLeastSquares
from rectools.dataset import Dataset
from rectools.models import ImplicitALSWrapperModel

K_RECOS = 40
RANDOM_STATE = 32
ITERATIONS = 10
FACTOR = 128
REGULARIZATION = 1
ALPHA = 1.0
FIT_FEATURES_TOGETHER = True


def main() -> None:
    """Main function to train the ALS model."""
    df_train = pd.read_csv("data/train.csv")
    df_features = pd.read_csv("data/cat_features.csv")
    df_features.rename(columns={"node": "item_id"}, inplace=True)

    model = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=FACTOR,
            regularization=REGULARIZATION,
            alpha=ALPHA,
            random_state=RANDOM_STATE,
            use_gpu=False,
            iterations=ITERATIONS,
        ),
        fit_features_together=FIT_FEATURES_TOGETHER,
    )

    dataset = Dataset.construct(
        interactions_df=df_train,
        item_features_df=df_features,
        cat_item_features=["category"],
    )

    model.fit(dataset)

    model.save("models/als_model.pkl")


if __name__ == "__main__":
    main()
