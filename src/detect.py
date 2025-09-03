# TODO: predictions are human-readable labels instead of numbers

import os
import joblib
import pandas as pd
from typing import Literal, Union
from features import prepare_features, load_processed

DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def load_model(dataset: DatasetName):
    """
    Load a trained model from /models/.
    """
    model_path = os.path.join(MODEL_DIR, f"{dataset}_rf.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"📂 Loading model: {model_path}")
    return joblib.load(model_path)


def detect(
    dataset: DatasetName,
    input_data: Union[str, pd.DataFrame],
    batch_size: int = None
) -> pd.DataFrame:
    """
    Run anomaly detection on new data.

    Args:
        dataset (str): Which dataset model to use ("CIC-IDS2017" or "UNSW-NB15")
        input_data (str | pd.DataFrame): CSV file path OR DataFrame
        batch_size (int): Optional, number of rows to process at once (for large files)

    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Load trained model
    model = load_model(dataset)

    # Load input
    if isinstance(input_data, str):
        print(f"📂 Reading input file: {input_data}")
        df = pd.read_csv(input_data, low_memory=False)
    else:
        df = input_data.copy()

    # Transform features
    X, _ = prepare_features(df, dataset)

    # Batch inference (optional)
    if batch_size:
        preds = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            preds.extend(model.predict(batch))
    else:
        preds = model.predict(X)

    results = df.copy()
    results["prediction"] = preds

    return results


if __name__ == "__main__":
    # Example: use processed dataset as a test input
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "CIC-IDS2017_clean.csv")
    predictions = detect("CIC-IDS2017", test_csv, batch_size=10000)

    print(predictions[["prediction"]].value_counts())
    print(predictions.head())
