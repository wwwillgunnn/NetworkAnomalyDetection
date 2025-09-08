# eda_tailored.py
import os
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ingest import load_dataset  # your repo loader

# Dataset-specific columns aligned with features.py (plus common IP fields)
COLS: Dict[str, dict] = {
    "CIC-IDS2017": {
        "num": [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
        ],
        "proto": ["Protocol"],           # will map to proto in features.py
        "label": ["Label"],              # will map to label in features.py
        "src": ["Source IP", "src_ip", "Src IP"],
        "dst": ["Destination IP", "dst_ip", "Dst IP"],
        "dur": ["Flow Duration"],
        "bytes_like": [
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Bytes", "bytes"
        ],
        "ts": ["Timestamp", "timestamp", "Time", "time"]
    },
    "UNSW-NB15": {
        "num": ["dur", "sbytes", "dbytes", "sttl", "dttl"],
        "proto": ["proto"],
        "label": ["attack_cat"],
        "src": ["srcip"],
        "dst": ["dstip"],
        "dur": ["dur"],
        "bytes_like": ["sbytes", "dbytes", "bytes"],
        "ts": ["stime", "time", "timestamp"]
    }
}

def _pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _ensure_dt(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    col = _pick(df, cands)
    if not col:
        return None
    if not np.issubdtype(df[col].dtype, np.datetime64):
        try:
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return col

def _save(outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.show()

def eda_run(dataset: str, outdir: str = "plots", sample: Optional[int] = 200_000):
    # 1) Load raw (keeps all cols, perfect for EDA)
    df = load_dataset(dataset)

    # Optional sampling for speed
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)

    spec = COLS[dataset]

    # 2) Class balance (binarize label: attack=1, benign/normal/background=0)
    lab_col = _pick(df, spec["label"])
    if lab_col:
        labels = df[lab_col].astype(str).str.lower().str.strip()
        y_bin = (~labels.isin({"benign", "normal", "background"})).astype(int)
        vc = y_bin.value_counts().reindex([0, 1]).fillna(0)
        plt.figure()
        plt.bar(["benign(0)", "attack(1)"], vc.values)
        plt.title(f"{dataset} – class balance"); plt.ylabel("count")
        _save(outdir, "class_balance")

    # 3) Missing values (top N)
    miss = df.isna().mean().sort_values(ascending=False).head(30)
    plt.figure()
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=90); plt.ylabel("fraction missing")
    plt.title(f"{dataset} – missing values (top 30)")
    _save(outdir, "missing_values")

    # 4) Numeric histograms for the dataset’s main features
    for c in spec["num"]:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            x = df[c].dropna().values
            plt.figure()
            plt.hist(x, bins=60)
            plt.title(f"{dataset} – {c}"); plt.xlabel(c); plt.ylabel("count")
            _save(outdir, f"hist_{c.replace(' ','_').lower()}")

    # 5) Protocol distribution (Protocol/proto)
    proto_col = _pick(df, spec["proto"])
    if proto_col:
        vc = df[proto_col].astype(str).value_counts()
        plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=90); plt.ylabel("count")
        plt.title(f"{dataset} – protocol distribution")
        _save(outdir, "protocol_distribution")

    # 6) Correlation heatmap over top-variance numeric columns (up to 20)
    num_cols = [c for c in spec["num"] if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols) >= 2:
        var_order = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
        chosen = var_order.head(20).index.tolist()
        corr = df[chosen].corr().values
        plt.figure(figsize=(6,5))
        im = plt.imshow(corr, aspect="auto", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(chosen)), chosen, rotation=90)
        plt.yticks(range(len(chosen)), chosen)
        plt.title(f"{dataset} – correlation (top variance)")
        _save(outdir, "correlation_heatmap")

    # 7) Bytes vs duration (log–log)
    dur_col = _pick(df, spec["dur"])
    bytes_col = _pick(df, spec["bytes_like"])
    if dur_col and bytes_col and np.issubdtype(df[dur_col].dtype, np.number) and np.issubdtype(df[bytes_col].dtype, np.number):
        D = df[[dur_col, bytes_col]].dropna()
        if len(D) > 50_000:
            D = D.sample(50_000, random_state=42)
        plt.figure()
        plt.scatter(D[dur_col].values + 1e-9, D[bytes_col].values + 1e-9, s=6, alpha=0.5)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel(dur_col); plt.ylabel(bytes_col)
        plt.title(f"{dataset} – bytes vs duration (log–log)")
        _save(outdir, "bytes_vs_duration")

    # 8) Top IPs & degree-ish view (if IP columns exist)
    src_col = _pick(df, spec["src"]); dst_col = _pick(df, spec["dst"])
    if src_col:
        top_src = df[src_col].astype(str).value_counts().head(20)[::-1]
        plt.figure(figsize=(6, max(3, len(top_src)*0.3)))
        plt.barh(top_src.index.astype(str), top_src.values)
        plt.xlabel("count"); plt.title(f"{dataset} – top source IPs")
        _save(outdir, "top_src_ips")
    if dst_col:
        top_dst = df[dst_col].astype(str).value_counts().head(20)[::-1]
        plt.figure(figsize=(6, max(3, len(top_dst)*0.3)))
        plt.barh(top_dst.index.astype(str), top_dst.values)
        plt.xlabel("count"); plt.title(f"{dataset} – top destination IPs")
        _save(outdir, "top_dst_ips")
    if src_col and dst_col:
        deg = pd.concat([df[src_col], df[dst_col]]).value_counts()
        ranks = np.arange(1, len(deg)+1)
        plt.figure()
        plt.loglog(ranks, np.sort(deg.values)[::-1])
        plt.xlabel("rank"); plt.ylabel("degree"); plt.title(f"{dataset} – degree rank (log–log)")
        _save(outdir, "degree_rank")

    # 9) Time series (events/bytes) if a timestamp-like column exists
    ts_col = _ensure_dt(df, spec["ts"])
    if ts_col:
        counts = df.set_index(ts_col).sort_index().resample("5T").size()
        plt.figure()
        counts.plot()
        plt.title(f"{dataset} – events per 5 minutes")
        plt.xlabel("time"); plt.ylabel("count")
        _save(outdir, "ts_counts")

        if bytes_col:
            series = df[[ts_col, bytes_col]].dropna().set_index(ts_col).sort_index()[bytes_col].resample("5T").sum()
            plt.figure()
            series.plot()
            plt.title(f"{dataset} – bytes per 5 minutes")
            plt.xlabel("time"); plt.ylabel("bytes")
            _save(outdir, "ts_bytes")

    # 10) PCA (2D) on the numeric feature set with label overlay if present
    try:
        from sklearn.decomposition import PCA
        nums = [c for c in num_cols if c in df.columns]
        if len(nums) >= 2:
            D = df[nums].dropna()
            if len(D) > 20_000:
                D = D.sample(20_000, random_state=42)
            X = (D - D.mean()) / (D.std(ddof=0) + 1e-9)
            Z = PCA(n_components=2, random_state=42).fit_transform(X.values)
            plt.figure()
            if lab_col is not None:
                y_local = (~df.loc[D.index, lab_col].astype(str).str.lower().str.strip()
                           .isin({"benign","normal","background"})).astype(int).values
                m0 = y_local == 0; m1 = ~m0
                plt.scatter(Z[m0,0], Z[m0,1], s=8, alpha=0.6, label="benign(0)")
                plt.scatter(Z[m1,0], Z[m1,1], s=8, alpha=0.6, label="attack(1)")
                plt.legend()
            else:
                plt.scatter(Z[:,0], Z[:,1], s=8, alpha=0.6)
            plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"{dataset} – PCA (2D)")
            _save(outdir, "pca_2d")
    except Exception as e:
        print(f"PCA skipped: {e}")

if __name__ == "__main__":
    # Examples:
    # python eda_tailored.py  (edit below to switch dataset)
    eda_run("CIC-IDS2017", outdir="plots_cic")
    eda_run("UNSW-NB15", outdir="plots_unsw")
