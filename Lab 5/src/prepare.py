"""
Stage 1: Prepare Data

Loads the California Housing dataset,
splits it into train and test datasets,
and saves them as CSV files.
"""

import yaml
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")


def load_dataset() -> pd.DataFrame:
    housing = fetch_california_housing(as_frame=True)

    df = housing.frame

    # Rename target column
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)

    return df


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    OUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"Train Shape : {train_df.shape}")
    print(f"Test Shape  : {test_df.shape}")

    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()