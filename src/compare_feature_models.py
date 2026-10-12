
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT / "data" / "processed" / "sensor_features.csv"
PROFILE_FILE = PROJECT / "data" / "raw" / "profile.txt"

# Load sensor features and cooler-condition labels
X = pd.read_csv(FEATURE_FILE)
profile = pd.read_csv(PROFILE_FILE, sep=r"\s+", header=None)
y = profile.iloc[:, 0]

if len(X) != len(profile):
    raise ValueError("Sensor data and profile row counts do not match.")

# Keep identical complete operating profiles in the same split
groups = profile.astype(str).agg("_".join, axis=1)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Fit the full model using training data only
full_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("Training full model...")
full_model.fit(X_train, y_train)

full_predictions = full_model.predict(X_test)
full_accuracy = accuracy_score(y_test, full_predictions)

# Select features using training data only
print("Selecting features from training data...")
importance = permutation_importance(
    full_model,
    X_train,
    y_train,
    scoring="accuracy",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance_mean": importance.importances_mean
}).sort_values("importance_mean", ascending=False)

top_features = importance_df.head(10)["feature"].tolist()

# Train the reduced model on the same training rows
reduced_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("Training reduced model...")
reduced_model.fit(X_train[top_features], y_train)

reduced_predictions = reduced_model.predict(X_test[top_features])
reduced_accuracy = accuracy_score(y_test, reduced_predictions)

# Save feature rankings and comparison results
figures_dir = PROJECT / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(
    figures_dir / "training_feature_importance.csv",
    index=False
)

results = pd.DataFrame([
    {
        "model": "Full model",
        "features": X.shape[1],
        "accuracy_percent": round(full_accuracy * 100, 2)
    },
    {
        "model": "Reduced model",
        "features": len(top_features),
        "accuracy_percent": round(reduced_accuracy * 100, 2)
    }
])

results.to_csv(
    figures_dir / "corrected_model_comparison.csv",
    index=False
)

print("\nTOP 10 FEATURES SELECTED FROM TRAINING DATA")
print(top_features)

print("\nCORRECTED MODEL COMPARISON")
print(results.to_string(index=False))

print("\nSaved results to:")
print(figures_dir / "corrected_model_comparison.csv")
