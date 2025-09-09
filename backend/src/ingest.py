import os
import pandas as pd
from typing import Literal

# Define allowed dataset names so function calls are type-checked.
DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

# Dynamically resolve project root (this avoids hardcoding paths).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "samples")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)  # ensure folder exists


def load_data(dataset: DatasetName, base_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    # Load raw CSV files for a dataset into a single pandas DataFrame (raw, uncleaned)
    # TODO: Potentially look into merging the two datasets into one

    dataset_path = os.path.join(base_path, dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_path}")

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)
        print(f"📂 Loading {file_path} ...")

        try:
            df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8 decode failed for {file_path}, falling back to latin-1")
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
            # ! Later in features.py, you explicitly convert columns to int, float, or keep as category

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"✅ Loaded {len(combined_df)} rows from {dataset}")
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on raw dataset:
      - Drop unnamed columns
      - Strip whitespace from column names
      - Drop duplicates
      - Replace NaNs with 0
    """
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df


def save_processed(df: pd.DataFrame, dataset: DatasetName, base_path: str = PROCESSED_DATA_PATH) -> str:
    """
    Save cleaned dataset to /data/processed/ as a single CSV.
    Returns path to saved file.
    """
    os.makedirs(base_path, exist_ok=True)
    out_path = os.path.join(base_path, f"{dataset}_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved processed dataset → {out_path}")
    return out_path


if __name__ == "__main__":
    # Example: process both datasets
    for ds in ["CIC-IDS2017", "UNSW-NB15"]:
        raw_df = load_data(ds)
        clean_df = clean_data(raw_df)
        save_processed(clean_df, ds)
        print(clean_df.head(), clean_df.shape)
