import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Literal, Tuple

DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

def prepare_features(df: pd.DataFrame, dataset: DatasetName) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw DataFrame into model-ready (X, y).

    Args:
        df (pd.DataFrame): Raw dataset from ingest.py
        dataset (str): "CIC-IDS2017" or "UNSW-NB15"

    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Encoded labels
    """

    if dataset == "CIC-IDS2017":
        # Example columns from CIC-IDS2017
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
        # Example columns from UNSW-NB15
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

    # Keep only the relevant columns that exist in df
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()

    # Standardize naming across datasets
    if dataset == "CIC-IDS2017":
        df.rename(columns={"Label": "label", "Protocol": "proto"}, inplace=True)
    elif dataset == "UNSW-NB15":
        df.rename(columns={"attack_cat": "label", "proto": "proto"}, inplace=True)

    # Handle categorical encoding
    label_enc = LabelEncoder()
    proto_enc = LabelEncoder()

    if "proto" in df.columns:
        df["proto"] = proto_enc.fit_transform(df["proto"].astype(str))

    y = label_enc.fit_transform(df["label"].astype(str))

    # Drop label from X
    X = df.drop(columns=["label"])

    # Scale numeric features (important for ML models like SVM, neural nets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, pd.Series(y, name="label")


if __name__ == "__main__":
    from ingest import load_dataset

    # Load CIC-IDS2017
    cic_df = load_dataset("CIC-IDS2017")
    X_cic, y_cic = prepare_features(cic_df, "CIC-IDS2017")
    print("CIC-IDS2017:", X_cic.shape, y_cic.shape)

    # Load UNSW-NB15
    unsw_df = load_dataset("UNSW-NB15")
    X_unsw, y_unsw = prepare_features(unsw_df, "UNSW-NB15")
    print("UNSW-NB15:", X_unsw.shape, y_unsw.shape)