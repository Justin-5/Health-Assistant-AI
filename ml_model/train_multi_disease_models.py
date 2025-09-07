import os
from pathlib import Path
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ML_DIR = Path(__file__).parent


def find_data_file(filename: str) -> Path:
    """Locate a data file by searching common base directories."""
    cwd = Path.cwd()
    bases: List[Path] = [
        cwd,
        cwd / "ml_model",
        ML_DIR,
        ML_DIR.parent,
    ]
    candidates = []
    for base in bases:
        candidates.append(base / filename)
        candidates.append(base / "ml_model" / filename)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Tried: " + ", ".join(str(p) for p in candidates))


def load_dataset(filename: str) -> pd.DataFrame:
    path = find_data_file(filename)
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def build_preprocess_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def train_and_serialize(
    dataset_file: str,
    target_col: str,
    model_name_prefix: str,
    target_candidates: Optional[List[str]] = None,
) -> None:
    print(f"\n=== Training for {model_name_prefix} ===")
    df = load_dataset(dataset_file)
    # Auto-detect target if needed
    actual_target = target_col
    if target_candidates:
        for c in target_candidates:
            if c in df.columns:
                actual_target = c
                break
    if actual_target not in df.columns:
        raise KeyError(
            f"None of target candidates found: {target_candidates}. Available: {list(df.columns)}")

    # Drop rows with missing target
    df = df.dropna(subset=[actual_target])
    X, y = split_features_target(df, actual_target)

    # Ensure binary target where appropriate (convert yes/no, strings to categorical codes)
    if y.dtype == object:
        y = y.astype("category").cat.codes

    # Train/test split with stratification if possible
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Build preprocessing and model pipeline
    preprocessor = build_preprocess_pipeline(X_train)
    model = LogisticRegression(max_iter=1000, n_jobs=None)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", model),
    ])

    # Fit
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({model_name_prefix}): {acc:.4f}")

    # Persist: save the full pipeline as model, and also save the fitted scaler separately
    # Extract the scaler fit on numeric columns to save separately, if present
    model_path = ML_DIR / f"{model_name_prefix}_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model pipeline to {model_path}")

    # Save a separate StandardScaler fitted on the full train set numeric columns
    # Note: The scaler is nested inside the ColumnTransformer; extract it for convenience
    scaler = None
    preprocess_step: ColumnTransformer = pipeline.named_steps["preprocess"]
    for name, trans, cols in preprocess_step.transformers_:
        if name == "num":
            scaler = trans.named_steps.get("scaler")
            break
    if scaler is not None:
        scaler_path = ML_DIR / f"{model_name_prefix}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    else:
        print("No numeric scaler found to serialize (no numeric columns detected).")


def main():
    # Diabetes
    train_and_serialize(
        dataset_file="diabetes_012_health_indicators_BRFSS2015.csv",
        target_col="Diabetes_012",
        model_name_prefix="diabetes",
    )

    # Heart disease (try multiple target names)
    train_and_serialize(
        dataset_file="heart.csv",
        target_col="HeartDisease",
        model_name_prefix="heart",
        target_candidates=["HeartDisease", "target", "Outcome"],
    )

    # Hypertension (try multiple target names)
    train_and_serialize(
        dataset_file="hypertension_dataset.csv",
        target_col="Hypertension",
        model_name_prefix="hypertension",
        target_candidates=["Hypertension",
                           "hypertension", "target", "Outcome"],
    )


if __name__ == "__main__":
    main()

