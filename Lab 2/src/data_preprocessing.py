"""
Stage 2: Data Preprocessing
-----------------------------
Reads the raw CSV, cleans column names, checks/handles missing values and
duplicates, and writes a processed CSV.

Input:
    data/raw/data.csv

Output:
    data/processed/data.csv
"""

import os
import pandas as pd


def load_raw_data(path: str = "data/raw/data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[data_preprocessing] Loaded raw data (shape={df.shape})")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    if before != after:
        print(f"[data_preprocessing] Dropped {before - after} duplicate rows")

    # Handle missing values
    if df.isnull().sum().sum() > 0:

        # Fill numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        print("[data_preprocessing] Missing values handled")

    # Target should be numeric for regression
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # Remove rows where target is missing
    df = df.dropna(subset=["target"])

    return df


def save_processed_data(df: pd.DataFrame, out_dir: str = "data/processed") -> None:
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "data.csv")

    df.to_csv(out_path, index=False)

    print(f"[data_preprocessing] Saved processed data -> {out_path} (shape={df.shape})")


def main():
    df = load_raw_data()
    df = clean_data(df)
    save_processed_data(df)


if __name__ == "__main__":
    main()