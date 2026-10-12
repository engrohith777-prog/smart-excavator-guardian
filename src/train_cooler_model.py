
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project paths
PROJECT = Path(__file__).resolve().parent.parent
FEATURE_FILE = PROJECT / "data" / "processed" / "sensor_features.csv"
PROFILE_FILE = PROJECT / "data" / "raw" / "profile.txt"
MODEL_FOLDER = PROJECT / "models"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Load sensor features and condition labels
X = pd.read_csv(FEATURE_FILE)

profile = pd.read_csv(
    PROFILE_FILE,
    sep=r"\s+",
    header=None,
    engine="python"
)

# Check that the cycle counts match
if len(X) != len(profile):
    raise ValueError("Feature and profile row counts do not match!")

# Column 0 contains the cooler-condition label
y = profile.iloc[:, 0]

print("Sensor cycles:", len(X))
print("Number of features:", X.shape[1])
print("\nCooler-condition counts:")
print(y.value_counts().sort_index())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the first machine-learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\nTest accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model and feature names
model_path = MODEL_FOLDER / "cooler_condition_model.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns),
        "target_name": "cooler_condition"
    },
    model_path
)

print("\nModel saved to:", model_path)
print("Training completed!")
