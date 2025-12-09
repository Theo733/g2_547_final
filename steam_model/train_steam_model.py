#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Project description (updated: binary classification version)
# =============================================================================
# This script trains a supervised learning model to predict whether a player
# will increase their playtime AFTER writing a Steam review (binary outcome).
#
# Data:
#   - Three Steam review datasets for specific games, stored as Excel files:
#       Cyberpunk 2077_steam_reviews_1091500.part000_partial.xlsx
#       Red Dead Redemption2_steam_reviews_1174180.part000_with_sent.xlsx
#       Witcher 3_steam_reviews_292030.part000_with_sent.xlsx
#
# Target variable (binary):
#   - author_playtime_forever: total playtime in minutes (measured later)
#   - author_playtime_at_review: playtime in minutes at review time
#   - We define the binary target:
#       target_increase = 1  if author_playtime_forever > author_playtime_at_review
#                        0  otherwise
#     In words: 1 if the player plays more after the review, 0 if they do not
#     increase their total minutes (no change).
#
# Main modeling choices:
#   1. We remove rows with missing playtime columns and rows where
#      delta_playtime = author_playtime_forever - author_playtime_at_review < 0
#      (these look like data inconsistency).
#   2. We still compute delta_playtime as an intermediate quantity for cleaning
#      and descriptive purposes, but the final target is binary (increase vs no
#      increase), not a continuous number of minutes.
#   3. We use HistGradientBoostingClassifier as the main model, with all
#      review-level features defined below. This allows non-linear interactions.
#   4. We split by author_steamid using GroupShuffleSplit to avoid
#      information leak between train/val/test, since one player may write
#      multiple reviews.
#
# This file has been adapted to:
#   - log metrics and the model to MLflow
#   - register the model in the MLflow Model Registry
#   - assume an MLflow tracking server is already running and configured
#     (e.g., via docker-compose + DO Spaces artifact store).
# =============================================================================

from dotenv import load_dotenv
load_dotenv()  # this loads variables from a .env file in the working directory

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sqlalchemy import create_engine

import mlflow
from mlflow.models import infer_signature


# ---------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Postgres connection URL – must be set in .env
# Example (DigitalOcean):
#   APP_DB_URL=postgresql://doadmin:<password>@<host>:25060/defaultdb?sslmode=require
DB_URL = os.getenv("APP_DB_URL")

# Excel files remain here, but are ONLY used by the loader script (not training).
DATA_FILES = [
    DATA_DIR / "Cyberpunk 2077_steam_reviews_1091500.part000_partial.xlsx",
    DATA_DIR / "Red Dead Redemption2_steam_reviews_1174180.part000_with_sent.xlsx",
    DATA_DIR / "Witcher 3_steam_reviews_292030.part000_with_sent.xlsx",
]

# Random seeds and split proportions
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # relative to remaining after test split

# ---------------------------------------------------------------------
# 1. Data loading helpers
# ---------------------------------------------------------------------


def load_one_file(path: Path) -> pd.DataFrame:
    """
    Load a single Excel file into a DataFrame.
    """
    print(f"Loading file: {path}")
    df = pd.read_excel(str(path))
    df["source_file"] = os.path.basename(str(path))
    return df


def load_all_files(file_list) -> pd.DataFrame:
    """
    Load and concatenate all Excel files in file_list.
    """
    dfs = []
    for p in file_list:
        df = load_one_file(p)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(file_list)} files.")
    return combined


# ---------------------------------------------------------------------
# 2. Data cleaning & feature engineering
# ---------------------------------------------------------------------


def prepare_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Steam review data and construct the binary target.

    Steps (high-level):
      - Keep only relevant columns.
      - Drop rows with missing or inconsistent playtime fields.
      - Create binary target: target_increase (1 if playtime goes up).
      - Some additional cleaning on author_steamid and other features.
    """

    df = df_raw.copy()

    # Ensure numeric fields
    numeric_cols = [
        "author_playtime_forever",
        "author_playtime_at_review",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Required numeric column {col} not found.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing playtime information
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)
    print(
        f"Dropped {before - after} rows with missing playtime; "
        f"remaining: {after}"
    )

    # Compute delta_playtime
    df["delta_playtime"] = (
        df["author_playtime_forever"] - df["author_playtime_at_review"]
    )

    # Drop negative deltas (data inconsistencies)
    before = len(df)
    df = df[df["delta_playtime"] >= 0]
    after = len(df)
    print(
        f"Dropped {before - after} rows with negative delta_playtime; "
        f"remaining: {after}"
    )

    # Define binary target
    df["target_increase"] = (
        df["author_playtime_forever"] > df["author_playtime_at_review"]
    ).astype(int)
    print(
        "Binary target distribution: "
        f"{df['target_increase'].value_counts(normalize=True).to_dict()}"
    )

    # Ensure author_steamid exists and is usable for grouping
    if "author_steamid" not in df.columns:
        raise ValueError("Column 'author_steamid' is required but not found.")
    df["author_steamid"] = df["author_steamid"].astype(str).str.strip()

    # Drop rows with missing author_steamid
    before_auth = len(df)
    df = df.replace({"": np.nan})
    df = df.dropna(subset=["author_steamid"])
    after_auth = len(df)
    print(
        f"Dropped {before_auth - after_auth} rows with missing author_steamid; "
        f"remaining: {after_auth}"
    )

    return df


# ---------------------------------------------------------------------
# 3. Group-based train/val/test split
# ---------------------------------------------------------------------


def group_train_val_test_split(
    df: pd.DataFrame,
    group_col: str = "author_steamid",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split df into train/val/test using GroupShuffleSplit on group_col so that
    no group (author_steamid) appears in more than one split.
    """

    # First split off test
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    groups = df[group_col]
    idx_trainval, idx_test = next(
        gss_test.split(df, groups=groups, y=None)
    )
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    print(
        f"Initial split: train+val = {len(df_trainval)}, "
        f"test = {len(df_test)}"
    )

    # Now split trainval into train and val
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state + 1
    )
    groups_trainval = df_trainval[group_col]
    idx_train, idx_val = next(
        gss_val.split(df_trainval, groups=groups_trainval, y=None)
    )

    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)

    print(
        f"Final splits: train = {len(df_train)}, "
        f"val = {len(df_val)}, test = {len(df_test)}"
    )

    return df_train, df_val, df_test


# ---------------------------------------------------------------------
# 4. Model + feature pipeline
# ---------------------------------------------------------------------


def build_model():
    """
    Build a sklearn Pipeline for binary classification using
    HistGradientBoostingClassifier and basic preprocessing.

    Returns:
      model: sklearn Pipeline
      feature_cols: list of feature column names
    """

    # Example feature selection: adjust to match your real columns
    numeric_features = [
        "author_playtime_at_review",
        "author_playtime_forever",
        # add other numeric features here
    ]

    categorical_features = [
        "voted_up",  # example boolean / categorical
        # add other categorical columns here
    ]

    feature_cols = numeric_features + categorical_features

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    return clf, feature_cols


# ---------------------------------------------------------------------
# 5. Evaluation helper
# ---------------------------------------------------------------------


def evaluate_classification(y_true, y_pred, split_name: str):
    """
    Print classification metrics:
      - Accuracy
      - F1-score for the positive class (label=1)
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    print(f"[{split_name}] Accuracy = {acc:.4f}, F1 (increase=1) = {f1:.4f}")


# ---------------------------------------------------------------------
# 6. Main training + MLflow logging
# ---------------------------------------------------------------------


def main():
    # 1) Load data from Postgres (Excel option disabled)
    if DB_URL is None:
        raise RuntimeError(
            "APP_DB_URL is not set. This script now requires a Postgres database. "
            "Set APP_DB_URL in your .env to your DigitalOcean Postgres URL."
        )

    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM steam_reviews_prepared", engine)
    print(f"Loaded {len(df)} rows from Postgres table 'steam_reviews_prepared'.")

    # 2) Split by author_steamid (group-based)
    df_train, df_val, df_test = group_train_val_test_split(
        df, group_col="author_steamid"
    )

    # 3) Build model and feature list
    model, feature_cols = build_model()

    # 4) Prepare X / y (target = target_increase)
    X_train = df_train[feature_cols]
    y_train = df_train["target_increase"].astype(int)

    X_val = df_val[feature_cols]
    y_val = df_val["target_increase"].astype(int)

    X_test = df_test[feature_cols]
    y_test = df_test["target_increase"].astype(int)

    # 5) MLflow run
    with mlflow.start_run():
        # Basic run metadata
        mlflow.set_tag("project", "steam_playtime_increase")
        mlflow.set_tag("model_type", "HistGradientBoostingClassifier")

        # Log simple configuration info
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "random_state": RANDOM_STATE,
            }
        )

        # 6) Fit model
        print("\nFitting model...")
        model.fit(X_train, y_train)

        # 7) Predictions on each split
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Probabilities if available
        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # Map decision_function outputs into [0, 1] for logging
            logits = model.decision_function(X_test)
            y_test_proba = 1 / (1 + np.exp(-logits))
        else:
            # Fallback: no probability interface; use zeros
            y_test_proba = np.zeros_like(y_test_pred, dtype=float)

        # 8) Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred, pos_label=1)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1)

        mlflow.log_metrics(
            {
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "test_accuracy": float(test_acc),
                "train_f1": float(train_f1),
                "val_f1": float(val_f1),
                "test_f1": float(test_f1),
            }
        )

        # 9) Evaluation prints
        print("Evaluation on binary target (increase=1, no increase=0):")
        evaluate_classification(y_train, y_train_pred, "Train")
        evaluate_classification(y_val, y_val_pred, "Validation")
        evaluate_classification(y_test, y_test_pred, "Test")

        print("\nDetailed classification report on Test set:")
        print(classification_report(y_test, y_test_pred))

        print("\nConfusion matrix on Test set:")
        print(confusion_matrix(y_test, y_test_pred))

        # 10) Save test predictions
        df_test_out = df_test.copy()
        df_test_out["target_increase_true"] = y_test
        df_test_out["target_increase_pred"] = y_test_pred
        df_test_out["target_increase_proba"] = y_test_proba

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "test_predictions_playtime_increase_binary.csv"
        df_test_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved test predictions to: {out_path}")

        # 11) Log model to MLflow (Model Registry + artifacts in Spaces)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="steam-playtime-model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name="steam_playtime_increase_model",
        )



if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
    )
    mlflow.set_experiment("steam_playtime_increase")

    main()
