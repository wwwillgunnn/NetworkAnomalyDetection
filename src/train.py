import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from ingest import load_dataset
from features import prepare_features

# Path to save trained models
#Testing push
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(dataset_name: str = "CIC-IDS2017"):
    """
    Train a baseline RandomForest classifier on the given dataset.
    Saves model into /models/<dataset>_rf.joblib
    """

    # Load and featurize
    print(f"🔹 Loading dataset: {dataset_name}")
    raw_df = load_dataset(dataset_name)
    X, y = prepare_features(raw_df, dataset_name)

    # Train/test split
    print("🔹 Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Model
    print("🔹 Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_rf.joblib")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to {model_path}")


if __name__ == "__main__":
    # Train on both datasets for testing
    train_model("CIC-IDS2017")
    train_model("UNSW-NB15")
