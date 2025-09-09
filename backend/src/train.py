import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

from features import load_processed, prepare_features
# from autoencoder import Autoencoder

DatasetName = Literal["CIC-IDS2017", "UNSW-NB15"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def plot_confusion(y_true, y_pred, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


# def train_autoencoder(X_train, X_test, y_test, dataset_name: str):
    # """Train a PyTorch Autoencoder for anomaly detection."""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Convert to PyTorch tensors
    # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    # X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    # train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor),
    #                           batch_size=128, shuffle=True)

    # # Model, loss, optimizer
    # model = Autoencoder(input_dim=X_train.shape[1]).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    # epochs = 20
    # model.train()
    # for epoch in range(epochs):
    #     total_loss = 0
    #     for batch_X, _ in train_loader:
    #         batch_X = batch_X.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(batch_X)
    #         loss = criterion(outputs, batch_X)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.6f}")

    # # Save model
    # model_path = os.path.join(MODEL_DIR, f"{dataset_name}_autoencoder.pth")
    # torch.save(model.state_dict(), model_path)
    # print(f"💾 Autoencoder model saved → {model_path}")

    # # Evaluation (reconstruction error threshold)
    # model.eval()
    # with torch.no_grad():
    #     reconstructed = model(X_test_tensor.to(device)).cpu().numpy()
    #     mse = np.mean(np.power(X_test.values - reconstructed, 2), axis=1)

    # threshold = np.percentile(mse, 95)  # 95th percentile
    # y_pred = (mse > threshold).astype(int)  # 1 = anomaly

    # acc = accuracy_score(y_test, y_pred)
    # print(f"✅ Autoencoder Accuracy: {acc:.4f}")
    # print(classification_report(y_test, y_pred))
    # plot_confusion(y_test, y_pred, f"{dataset_name} - Autoencoder", os.path.join(MODEL_DIR, f"{dataset_name}_autoencoder_cm.png"))


def train_models(dataset_name: DatasetName):
    """Train RF, IsolationForest, Autoencoder + save scaler."""
    print(f"🔹 Loading processed dataset: {dataset_name}")
    df = load_processed(dataset_name)
    X, y = prepare_features(df, dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Save StandardScaler
    scaler = StandardScaler().fit(X_train)
    scaler_path = os.path.join(MODEL_DIR, f"{dataset_name}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # --- RandomForest ---
    # IsolationForest is now treated as an unsupervised anomaly detector. (-1 → anomaly, 1 → normal → mapped to y labels).
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("✅ RandomForest", classification_report(y_test, rf_pred))
    joblib.dump(rf, os.path.join(MODEL_DIR, f"{dataset_name}_rf.joblib"))
    plot_confusion(y_test, rf_pred, f"{dataset_name} - RF", os.path.join(MODEL_DIR, f"{dataset_name}_rf_cm.png"))

    # --- IsolationForest ---
    ifm = IsolationForest(n_estimators=200, contamination=0.1, max_features=0.7, random_state=42)
    ifm.fit(X_train)
    ifm_pred = np.where(ifm.predict(X_test) == -1, 1, 0)
    print("✅ IsolationForest", classification_report(y_test, ifm_pred))
    joblib.dump(ifm, os.path.join(MODEL_DIR, f"{dataset_name}_iforest.joblib"))
    plot_confusion(y_test, ifm_pred, f"{dataset_name} - IF", os.path.join(MODEL_DIR, f"{dataset_name}_iforest_cm.png"))

    # --- Autoencoder ---
    # train_autoencoder(X_train, X_test, y_test, dataset_name)


if __name__ == "__main__":
    for ds in ["CIC-IDS2017", "UNSW-NB15"]:
        train_models(ds)
