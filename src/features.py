# src/features.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Literal, Tuple

DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

def prepare_features(df: pd.DataFrame, dataset: DatasetName) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw DataFrame into model-ready (X, y), without scaling.
    Scaling will be handled in train.py for consistency.
    """

    if dataset == "CIC-IDS2017":
        cols_to_keep = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Protocol",
            "Label"
        ]

    elif dataset == "UNSW-NB15":
        cols_to_keep = [
            "dur",
            "sbytes",
            "dbytes",
            "sttl",
            "dttl",
            "proto",
            "attack_cat"
        ]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Keep only available cols
    available_cols = [c for c in cols_to_keep if c in df.columns]
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

    # Handle categorical encoding
    label_enc = LabelEncoder()
    proto_enc = LabelEncoder()

    if "proto" in df.columns:
        df["proto"] = proto_enc.fit_transform(df["proto"].astype(str))

    # Encode y labels (Normal/Attack → 0/1)
    y = label_enc.fit_transform(df["label"].astype(str))

    # Drop label from features
    X = df.drop(columns=["label"])

    return X, pd.Series(y, name="label")
