"""This script prepares the data for the model."""

from datetime import timedelta

import pandas as pd

EVAL_DAYS_TRESHOLD = 14


def main() -> None:
    """Main function to prepare the data for the model."""
    df_clickstream = pd.read_csv("data/fclickstream.csv")
    df_event = pd.read_csv("data/event.csv")

    df_clickstream["event_date"] = pd.to_datetime(df_clickstream["event_date"])

    treshhold = df_clickstream["event_date"].max() - timedelta(days=EVAL_DAYS_TRESHOLD)

    df_train = df_clickstream[df_clickstream["event_date"] <= treshhold]
    df_eval = df_clickstream[df_clickstream["event_date"] > treshhold][
        ["cookie", "node", "event"]
    ]

    df_eval = df_eval.merge(
        df_train[["cookie", "node"]], on=["cookie", "node"], how="left", indicator=True
    )
    df_eval = df_eval[df_eval["_merge"] == "left_only"].drop("_merge", axis=1)

    contact_events = df_event[df_event["is_contact"] == 1]["event"].unique()
    df_eval = df_eval[df_eval["event"].isin(contact_events)]

    df_eval = df_eval[df_eval["cookie"].isin(df_train["cookie"].unique())]
    df_eval = df_eval[df_eval["node"].isin(df_train["node"].unique())]

    df_eval = df_eval.drop_duplicates(["cookie", "node"])
    df_eval.rename(
        columns={"cookie": "user_id", "node": "item_id", "event_date": "datetime"},
        inplace=True,
    )

    df_train.rename(
        columns={"cookie": "user_id", "node": "item_id", "event_date": "datetime"},
        inplace=True,
    )
    df_train["weight"] = 1.0
    df_train["datetime"] = pd.to_datetime(df_train["datetime"])
    df_train = df_train[["user_id", "item_id", "datetime", "weight"]]

    df_train.to_csv("data/train.csv", index=False)
    df_eval.to_csv("data/eval.csv", index=False)


if __name__ == "__main__":
    main()
