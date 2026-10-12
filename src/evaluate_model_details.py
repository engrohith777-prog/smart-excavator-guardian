
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

# Project paths
PROJECT = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT / "data" / "processed" / "sensor_features.csv"
PROFILE_FILE = PROJECT / "data" / "raw" / "profile.txt"
OUTPUT_DIR = PROJECT / "reports" / "figures"

# Check required files
if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

if not PROFILE_FILE.exists():
    raise FileNotFoundError(f"Profile file not found: {PROFILE_FILE}")

# Load data
X = pd.read_csv(FEATURE_FILE)
profile = pd.read_csv(
    PROFILE_FILE,
    sep=r"\s+",
    header=None,
)

if len(X) != len(profile):
    raise ValueError(
        f"Row mismatch: {len(X)} sensor rows, "
        f"{len(profile)} profile rows."
    )

# Cooler-condition labels
y = profile.iloc[:, 0]

if y.isna().any():
    raise ValueError("Missing cooler-condition labels found.")

expected_labels = [3, 20, 100]
if sorted(y.unique().tolist()) != expected_labels:
    raise ValueError(
        f"Unexpected cooler-condition labels: {sorted(y.unique().tolist())}"
    )

# Keep identical complete operating profiles in the same split
groups = profile.astype(str).agg("_".join, axis=1)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("TRAINING AND TEST DATA")
print("Training cycles:", len(X_train))
print("Testing cycles:", len(X_test))
print("Sensor features:", X.shape[1])

# Train the full model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

print("\nTraining full model...")
model.fit(X_train, y_train)

# Predict test conditions
predictions = model.predict(X_test)

# Calculate accuracy and detailed metrics
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=expected_labels,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=expected_labels,
)

matrix_df = pd.DataFrame(
    matrix,
    index=[f"Actual_{label}" for label in expected_labels],
    columns=[f"Predicted_{label}" for label in expected_labels],
)

# Display results
print("\nACCURACY")
print(f"{accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        labels=expected_labels,
        zero_division=0,
    )
)

print("\nCONFUSION MATRIX")
print("Rows = actual condition; columns = predicted condition")
print(matrix_df)

# Save results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report_path = OUTPUT_DIR / "cooler_classification_report.csv"
matrix_path = OUTPUT_DIR / "cooler_confusion_matrix.csv"

report_df.to_csv(report_path)
matrix_df.to_csv(matrix_path)

print("\nFiles saved successfully:")
print(report_path)
print(matrix_path)
