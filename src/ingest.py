import os
import pandas as pd
from typing import Literal

# Define allowed dataset names so function calls are type-checked.
DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

# Dynamically resolve project root (this avoids hardcoding paths).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the /data/samples folder inside project root.
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "samples")

def load_dataset(dataset: DatasetName, base_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the datasets from /data/samples into a single pandas DataFrame.

    Args:
        dataset (str): Dataset name ("CIC-IDS2017" or "UNSW-NB15")
        base_path (str): Path to data directory.

    Returns:
        pd.DataFrame: Combined dataset.
    """

    # Construct the dataset folder path (e.g., data/samples/CIC-IDS2017).
    dataset_path = os.path.join(base_path, dataset)

    # Check if folder exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # List all CSV files
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_path}")

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)
        print(f"📂 Loading {file_path} ...")

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8 decode failed for {file_path}, falling back to latin-1")
            df = pd.read_csv(file_path, encoding="latin-1")

        # Drop unnamed index cols if present
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Basic cleanup
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.fillna(0)  # Replace NaNs with 0 for now

    print(f"✅ Loaded {len(combined_df)} rows from {dataset}")
    return combined_df


if __name__ == "__main__":
    # Sanity check: load a few rows
    cic_df = load_dataset("CIC-IDS2017")
    print(cic_df.head(), cic_df.shape)

    unsw_df = load_dataset("UNSW-NB15")
    print(unsw_df.head(), unsw_df.shape)
