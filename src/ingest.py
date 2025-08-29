import os
import pandas as pd
from typing import Literal

# Define allowed dataset names so function calls are type-checked.
# This way, if someone tries load_dataset("XYZ"), it will throw a typing error.
DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

# Dynamically resolve project root (this avoids hardcoding paths).
# __file__ = current file path (/src/ingest.py).
# dirname(dirname(...)) → one level above /src/, i.e. project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the /data/samples folder inside project root.
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "samples")

def load_dataset(dataset: DatasetName, base_path: str = DATA_PATH) -> pd.DataFrame:
    """
    This module will load the datasets from /data/samples into a single pandas DataFrame.

    Args:
        dataset (str): Dataset name ("CIC-IDS2017" or "UNSW-NB15")
        base_path (str): Path to data directory.

    Returns:
        pd.DataFrame: Combined dataset.
    """

    # Construct the dataset folder path (e.g., data/samples/CIC-IDS2017).
    dataset_path = os.path.join(base_path, dataset)

    # Check if folder actually exists, else raise error.
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # List all CSV files in the dataset folder.
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_path}")

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)
        print(f"📂 Loading {file_path} ...")
        df = pd.read_csv(file_path)

        # Drop unnamed index cols if present
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

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

# ! Later, extract a common subset of features (e.g., flow duration, packet count, bytes, protocol) from both datasets and see if a merged dataset improves performance. or train models sepeatley them ensemble



# Random chat code (prod ready apparently)
# from __future__ import annotations
# import os
# import glob
# import logging
# from pathlib import Path
# from typing import Dict, Iterable, Iterator, List, Literal, Optional
#
# import pandas as pd
#
# # ========= Logging =========
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )
# logger = logging.getLogger("ingest")
#
# DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]
#
# # ========= Unified schema =========
# # We'll normalize each dataset chunk to these columns when available.
# UNIFIED_COLUMNS = [
#     "timestamp",      # pd.Timestamp or str
#     "src_ip",
#     "dst_ip",
#     "src_port",
#     "dst_port",
#     "protocol",       # string or int
#     "bytes_in",       # optional depending on dataset
#     "bytes_out",      # optional depending on dataset
#     "bytes_total",
#     "packets_in",     # optional
#     "packets_out",    # optional
#     "packets_total",
#     "duration",       # seconds (float)
#     "label",          # original label string
#     "label_bin"       # 0 = benign/normal, 1 = attack/anomaly
# ]
#
# # ========= Dataset-specific mappings =========
# # Note: These datasets have varied column names. We map common fields to unified names.
# # You can extend these as needed once you inspect your exact CSV headers.
#
# CIC_MAP: Dict[str, str] = {
#     # Typical CIC-IDS2017 headers (may vary by split). Update once you confirm actual cols.
#     "Timestamp": "timestamp",
#     "Flow Duration": "duration",                # often in microseconds; we convert to seconds later
#     "Source IP": "src_ip",
#     "Destination IP": "dst_ip",
#     "Source Port": "src_port",
#     "Destination Port": "dst_port",
#     "Protocol": "protocol",
#     "Total Fwd Packets": "packets_out",         # forward = src->dst
#     "Total Backward Packets": "packets_in",     # backward = dst->src
#     "Total Length of Fwd Packets": "bytes_out",
#     "Total Length of Bwd Packets": "bytes_in",
#     "Label": "label",
# }
#
# UNSW_MAP: Dict[str, str] = {
#     # UNSW-NB15 common headers (e.g., from CSVs generated from the .arff):
#     "srcip": "src_ip",
#     "sport": "src_port",
#     "dstip": "dst_ip",
#     "dsport": "dst_port",
#     "proto": "protocol",
#     "state": "state",                 # we may ignore or keep as protocol metadata
#     "dur": "duration",
#     "sbytes": "bytes_out",
#     "dbytes": "bytes_in",
#     "spkts": "packets_out",
#     "dpkts": "packets_in",
#     "label": "label",                 # often 0/1 or "Normal"/"Attack" depending on split
#     "attack_cat": "attack_cat",       # optional category; we’ll fold into label if present
#     "stime": "timestamp",             # start time (epoch)
# }
#
# # Labels that indicate benign for CIC-IDS2017 (adjust if your files differ)
# CIC_BENIGN_TAGS = {"BENIGN", "Benign", "Normal", "CLEAN", "Non-Tor"}  # extend if needed
#
# # ========= Helpers =========
#
# def _list_csv_files(dataset_path: Path) -> List[Path]:
#     # Accept nested CSVs as well
#     patterns = [str(dataset_path / "*.csv"), str(dataset_path / "**" / "*.csv")]
#     files: List[Path] = []
#     for pat in patterns:
#         files.extend(Path(p).resolve() for p in glob.glob(pat, recursive=True))
#     unique_files = sorted({f for f in files if f.is_file()})
#     return unique_files
#
# def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
#     # Memory-friendly numeric downcast
#     for col in df.select_dtypes(include=["int64", "int32"]).columns:
#         df[col] = pd.to_numeric(df[col], downcast="integer")
#     for col in df.select_dtypes(include=["float64", "float32"]).columns:
#         df[col] = pd.to_numeric(df[col], downcast="float")
#     return df
#
# def _normalize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
#     # Derive totals where possible
#     if "bytes_total" not in df.columns:
#         if {"bytes_in", "bytes_out"}.issubset(df.columns):
#             df["bytes_total"] = df["bytes_in"].fillna(0) + df["bytes_out"].fillna(0)
#         elif "bytes_total" not in df.columns:
#             df["bytes_total"] = pd.NA
#
#     if "packets_total" not in df.columns:
#         if {"packets_in", "packets_out"}.issubset(df.columns):
#             df["packets_total"] = df["packets_in"].fillna(0) + df["packets_out"].fillna(0)
#         else:
#             df["packets_total"] = pd.NA
#
#     # Duration normalization: CIC often in microseconds; UNSW already seconds
#     if "duration" in df.columns:
#         # Heuristic: if max duration > 1e6, assume micros
#         try:
#             max_dur = pd.to_numeric(df["duration"], errors="coerce").dropna().max()
#             if pd.notna(max_dur) and max_dur and max_dur > 1e6:
#                 df["duration"] = pd.to_numeric(df["duration"], errors="coerce") / 1_000_000.0
#         except Exception:
#             pass
#
#     # Timestamp parsing (try best-effort)
#     if "timestamp" in df.columns:
#         # If numeric epoch -> convert; else try parse datetime strings
#         if pd.api.types.is_numeric_dtype(df["timestamp"]):
#             df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
#         else:
#             df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#
#     # Ensure ports are numeric where possible
#     for pcol in ("src_port", "dst_port"):
#         if pcol in df.columns:
#             df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
#
#     return df
#
# def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
#     # Ensure all unified columns exist (fill missing with NA)
#     for col in UNIFIED_COLUMNS:
#         if col not in df.columns:
#             df[col] = pd.NA
#     # Column order for consistency
#     df = df[UNIFIED_COLUMNS]
#     return _downcast_numeric(df)
#
# def _label_to_binary(label: Optional[str]) -> Optional[int]:
#     if label is None or (isinstance(label, float) and pd.isna(label)):
#         return None
#     s = str(label).strip()
#     if s.isdigit():
#         # Some UNSW exports already have 0/1
#         return int(s)
#     # Normalize common benign/attack conventions
#     if s in CIC_BENIGN_TAGS or s.lower() in {"benign", "normal"}:
#         return 0
#     return 1  # treat anything else as attack/anomaly by default
#
# # ========= Dataset normalizers =========
#
# def _normalize_cic(df: pd.DataFrame) -> pd.DataFrame:
#     # Map known CIC columns -> unified names
#     rename_map = {k: v for k, v in CIC_MAP.items() if k in df.columns}
#     df = df.rename(columns=rename_map)
#
#     # Fill bytes/packets totals if only one side is present
#     df = _normalize_common_fields(df)
#
#     # Label binarization
#     if "label" in df.columns:
#         df["label_bin"] = df["label"].apply(_label_to_binary)
#
#     return _finalize_schema(df)
#
# def _normalize_unsw(df: pd.DataFrame) -> pd.DataFrame:
#     rename_map = {k: v for k, v in UNSW_MAP.items() if k in df.columns}
#     df = df.rename(columns=rename_map)
#
#     # Prefer attack_cat if present for richer label text
#     if "attack_cat" in df.columns and "label" in df.columns:
#         # If label indicates 0/1 and attack_cat has category, build a composite text label
#         def _mk_label(row):
#             lab = row.get("label")
#             cat = row.get("attack_cat")
#             if pd.isna(cat) or str(cat).strip().lower() in {"nan", "", "na"}:
#                 return "Normal" if str(lab).strip() in {"0", "normal"} else "Attack"
#             if str(lab).strip() in {"0", "normal"}:
#                 return "Normal"
#             return str(cat)
#         df["label"] = df.apply(_mk_label, axis=1)
#
#     # Normalize shared fields
#     df = _normalize_common_fields(df)
#
#     # Label binarization
#     if "label" in df.columns:
#         df["label_bin"] = df["label"].apply(_label_to_binary)
#
#     return _finalize_schema(df)
#
# # ========= Public API =========
#
# def iter_dataset_chunks(
#     dataset: DatasetName,
#     base_path: str = "data/samples",
#     chunksize: int = 100_000,
#     low_memory: bool = True,
#     assume_missing: bool = True,
#     drop_chunk_duplicates: bool = True,
# ) -> Iterator[pd.DataFrame]:
#     """
#     Stream dataset rows in chunks, normalized to a unified schema.
#
#     Yields:
#         pd.DataFrame with columns UNIFIED_COLUMNS (missing values as NA)
#     """
#     dataset_path = Path(base_path) / dataset
#     if not dataset_path.exists():
#         raise FileNotFoundError(f"Dataset not found: {dataset_path}")
#
#     files = _list_csv_files(dataset_path)
#     if not files:
#         raise ValueError(f"No CSV files found under {dataset_path}")
#
#     logger.info("Found %d CSV file(s) for %s", len(files), dataset)
#
#     normalizer = _normalize_cic if dataset == "CIC-IDS2017" else _normalize_unsw
#
#     for fpath in files:
#         logger.info("Reading %s (chunksize=%d)", fpath, chunksize)
#         # NOTE: We don't predeclare dtypes here because columns vary across files.
#         # For production, consider profiling headers and setting an explicit dtype map.
#         reader = pd.read_csv(
#             fpath,
#             chunksize=chunksize,
#             low_memory=low_memory
#         )
#         for i, chunk in enumerate(reader):
#             if drop_chunk_duplicates:
#                 chunk = chunk.drop_duplicates()
#
#             # Drop unnamed index columns
#             chunk = chunk.loc[:, ~chunk.columns.str.match(r"^Unnamed")]
#
#             # Normalize per dataset
#             try:
#                 norm = normalizer(chunk)
#             except Exception as e:
#                 logger.exception("Failed to normalize chunk %s part %d: %s", fpath, i, e)
#                 continue
#
#             # Basic NA handling (keep NA; downstream can impute; but ensure label_bin int where possible)
#             yield norm
#
# def head(
#     dataset: DatasetName,
#     n: int = 5,
#     **kwargs
# ) -> pd.DataFrame:
#     """
#     Convenience: fetch the first n rows across streamed chunks (for debugging).
#     """
#     frames: List[pd.DataFrame] = []
#     acc = 0
#     for chunk in iter_dataset_chunks(dataset, **kwargs):
#         take = min(n - acc, len(chunk))
#         if take > 0:
#             frames.append(chunk.iloc[:take].copy())
#             acc += take
#         if acc >= n:
#             break
#     return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=UNIFIED_COLUMNS)
#
# # ========= CLI =========
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Stream & normalize dataset chunks.")
#     parser.add_argument("--dataset", type=str, choices=["CIC-IDS2017", "UNSW-NB15"], required=True)
#     parser.add_argument("--base_path", type=str, default="data/samples")
#     parser.add_argument("--chunksize", type=int, default=100_000)
#     parser.add_argument("--preview", type=int, default=10, help="Preview N rows and exit.")
#     args = parser.parse_args()
#
#     df = head(args.dataset, n=args.preview, base_path=args.base_path, chunksize=args.chunksize)
#     print(df.head(len(df)))
#     print(f"\nPreviewed {len(df)} rows with columns:\n{list(df.columns)}")
