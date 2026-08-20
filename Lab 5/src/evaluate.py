"""
Stage 3: Evaluate

Evaluates the trained regression model using
MAE, RMSE and R² Score.

If the R² Score is below the threshold,
the pipeline fails.
"""

import sys
import json
import yaml
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    # Load trained model
    model = joblib.load("model/model.joblib")

    # Load test data
    test_df = pd.read_csv("data/test.csv")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Predictions
    predictions = model.predict(X_test)

    # Regression Metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions),
        "R2_Score": r2_score(y_test, predictions),
    }

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))

    # Quality Gate
    if metrics["R2_Score"] < params["min_r2_score"]:
        print(
            f"\nFAIL: R² Score {metrics['R2_Score']:.4f} "
            f"is below the threshold of {params['min_r2_score']}"
        )
        sys.exit(1)

    print("\nPASS: Model cleared the quality gate.")


if __name__ == "__main__":
    main()