"""
Stage 4: Register

Uploads the trained regression model,
feature names, and model card
to Hugging Face Hub.

Runs only after the evaluation
quality gate has passed.
"""

import os
import json
from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ["HF_REPO_ID"]
HF_TOKEN = os.environ["HF_TOKEN"]


def build_model_card(metrics: dict) -> str:
    lines = [
        "---",
        "tags:",
        "- sklearn",
        "- random-forest",
        "- regression",
        "- mlops",
        "---",
        "",
        "# California Housing Price Prediction",
        "",
        "This model predicts California housing prices using a Random Forest Regressor.",
        "",
        "## Model Information",
        "",
        "- Framework: scikit-learn",
        "- Algorithm: RandomForestRegressor",
        "- Dataset: California Housing Dataset",
        "",
        "## Evaluation Metrics",
        "",
    ]

    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value:.4f}")

    lines.extend([
        "",
        "## Deployment",
        "",
        "This model was automatically trained, evaluated and deployed",
        "using a GitHub Actions CI/CD pipeline.",
    ])

    return "\n".join(lines)


def main():

    with open("metrics.json") as f:
        metrics = json.load(f)

    api = HfApi(token=HF_TOKEN)

    create_repo(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        exist_ok=True,
    )

    with open("model/README.md", "w") as f:
        f.write(build_model_card(metrics))

    files = [
        "model/model.joblib",
        "model/features.json",
        "model/README.md",
    ]

    for file in files:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=os.path.basename(file),
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )

    print("=" * 60)
    print("Model uploaded successfully!")
    print(f"https://huggingface.co/{REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()