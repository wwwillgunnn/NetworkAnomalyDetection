"""
features.py
-----------
Helpers to load (raw/processed) datasets and transform them into model-ready
features (X, y) for CIC-IDS2017 and UNSW-NB15.

This module intentionally reuses paths + functions from ingest.py to avoid drift.
"""
from __future__ import annotations
import os
from typing import Literal, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Reuse definitions from ingest.py
from ingest import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    load_data,        # raw CSV concatenation
    clean_data,       # basic cleaning step
    save_processed,   # write /data/processed/<dataset>_clean.csv
)


DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]
__all__ = ["load_dataset", "load_processed", "prepare_features"]


def load_dataset(dataset: DatasetName, base_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the RAW dataset by concatenating CSVs under data/samples/<dataset>.
    Returns an uncleaned DataFrame (may contain duplicates/NaNs/unnamed columns).
    """
    return load_data(dataset, base_path=base_path)


def load_processed(
    dataset: DatasetName,
    *,
    prefer_cache: bool = True,
    processed_path: str = PROCESSED_DATA_PATH,
    raw_path: str = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the CLEAN dataset:
      - If a cached processed CSV exists, read and return it.
      - Otherwise, load raw CSVs, clean them, cache to /data/processed, and return.

    Args:
        dataset: "CIC-IDS2017" | "UNSW-NB15"
        prefer_cache: If True, use cached processed CSV when available.
        processed_path: Folder for processed CSVs.
        raw_path: Folder for raw dataset CSVs.

    Returns:
        pd.DataFrame
    """
    os.makedirs(processed_path, exist_ok=True)
    cache_file = os.path.join(processed_path, f"{dataset}_clean.csv")

    if prefer_cache and os.path.exists(cache_file):
        print(f"📦 Loading cached processed dataset → {cache_file}")
        return pd.read_csv(cache_file, low_memory=False)

    # Build from raw
    raw_df = load_data(dataset, base_path=raw_path)
    cleaned = clean_data(raw_df)

    # Save and return
    save_processed(cleaned, dataset, base_path=processed_path)
    return cleaned


def prepare_features(df: pd.DataFrame, dataset: DatasetName) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform a (raw or cleaned) DataFrame into model-ready (X, y) without scaling.
    Scaling is handled later (e.g., in train.py), to keep one scaler per dataset split.

    Rules:
      - Subset to a small, stable feature set per dataset.
      - Normalize label column name to 'label'.
      - Normalize protocol column name to 'proto'.
      - Encode 'proto' if non-numeric; leave it as-is if already numeric.
      - Encode y with LabelEncoder.
    """
    if dataset == "CIC-IDS2017":
        cols_to_keep = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Protocol",
            "Label",
        ]
    elif dataset == "UNSW-NB15":
        cols_to_keep = [
            "dur",
            "sbytes",
            "dbytes",
            "sttl",
            "dttl",
            "proto",
            "attack_cat",
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Keep only available columns (print a gentle warning if some are missing)
    available_cols = [c for c in cols_to_keep if c in df.columns]
    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        print(f"⚠️ Warning: missing expected columns for {dataset}: {missing}")
    df = df[available_cols].copy()

    # Normalize label column name
    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)
    elif "attack_cat" in df.columns:
        df.rename(columns={"attack_cat": "label"}, inplace=True)
    elif "label" not in df.columns:
        raise KeyError("No label column found in dataset!")

    # Normalize protocol column name
    if "Protocol" in df.columns:
        df.rename(columns={"Protocol": "proto"}, inplace=True)

    # Encode proto ONLY if non-numeric (CIC protocol may be numeric already)
    if "proto" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["proto"]):
            proto_enc = LabelEncoder()
            df["proto"] = proto_enc.fit_transform(df["proto"].astype(str))

    # Encode y labels (e.g., BENIGN/Normal vs attacks)
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["label"].astype(str))

    # Drop label from features
    X = df.drop(columns=["label"])
    return X, pd.Series(y, name="label")
