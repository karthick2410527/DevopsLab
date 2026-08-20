"""
Stage 2: Train

Trains a Random Forest Regressor on the training dataset
and saves the trained model.
"""

import yaml
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = Path("model")


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    MODEL_DIR.mkdir(exist_ok=True)

    # Load training data
    train_df = pd.read_csv("data/train.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    # Create Regression Model
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )

    # Train Model
    model.fit(X_train, y_train)

    # Save Model
    joblib.dump(model, MODEL_DIR / "model.joblib")

    # Save Feature Names
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(list(X_train.columns), f, indent=4)

    print("=======================================")
    print("Model trained successfully!")
    print("Model saved to model/model.joblib")
    print("=======================================")


if __name__ == "__main__":
    main()