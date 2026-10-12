
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT / "data" / "processed" / "sensor_features.csv"
)
PROFILE_FILE = PROJECT / "data" / "raw" / "profile.txt"
OUTPUT_FOLDER = PROJECT / "reports" / "figures"

# Check that required files exist
if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

if not PROFILE_FILE.exists():
    raise FileNotFoundError(f"Profile file not found: {PROFILE_FILE}")

# --------------------------------------------------
# 2. Load sensor features and labels
# --------------------------------------------------
X = pd.read_csv(FEATURE_FILE)

profile = pd.read_csv(
    PROFILE_FILE,
    sep=r"\s+",
    header=None,
    engine="python"
)

if len(X) != len(profile):
    raise ValueError(
        f"Row mismatch: {len(X)} features but "
        f"{len(profile)} profile labels."
    )

if X.isnull().any().any():
    raise ValueError("Feature data contains missing values.")

if not X.select_dtypes(include="number").shape[1] == X.shape[1]:
    raise ValueError("All sensor features must be numeric.")

# Cooler-condition target
y = profile.iloc[:, 0]

# Group identical complete operating profiles together
groups = profile.astype(str).agg("_".join, axis=1)

print("Sensor cycles:", len(X))
print("Number of sensor features:", X.shape[1])
print("Cooler-condition classes:", sorted(y.unique()))

# --------------------------------------------------
# 3. Split data by operating profile
# --------------------------------------------------
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining cycles:", len(train_idx))
print("Testing cycles:", len(test_idx))
print(
    "Unique test profiles:",
    groups.iloc[test_idx].nunique()
)

# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Calculate permutation importance
# --------------------------------------------------
print("Calculating feature importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1
)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values(
    "importance_mean",
    ascending=False
).reset_index(drop=True)

# --------------------------------------------------
# 6. Print the results
# --------------------------------------------------
print("\nTop 15 important features:")
print(
    importance.head(15).to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

zero_count = (importance["importance_mean"] == 0).sum()
negative_count = (importance["importance_mean"] < 0).sum()
positive_count = (importance["importance_mean"] > 0).sum()

print("\nImportance summary:")
print("Total features:", len(importance))
print("Positive importance:", positive_count)
print("Zero importance:", zero_count)
print("Negative importance:", negative_count)

print(
    "\nNote: negative importance means shuffling a feature "
    "increased accuracy on this test set. This can occur "
    "because of sampling variability or noisy features."
)

# --------------------------------------------------
# 7. Save importance results
# --------------------------------------------------
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT_FOLDER / "cooler_feature_importance.csv"
importance.to_csv(csv_path, index=False)

# --------------------------------------------------
# 8. Create a professional chart
# --------------------------------------------------
top_features = (
    importance.head(15)
    .sort_values("importance_mean", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"],
    capsize=3
)

ax.axvline(0, linewidth=1)
ax.set_xlabel("Change in accuracy after feature shuffling")
ax.set_ylabel("Sensor feature")
ax.set_title("Cooler-Condition Model: Feature Importance")
ax.grid(axis="x", linestyle="--", alpha=0.4)

fig.tight_layout()

chart_path = OUTPUT_FOLDER / "cooler_feature_importance.png"
fig.savefig(chart_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nImportance table saved to:", csv_path)
print("Chart saved to:", chart_path)
print("\nFeature analysis completed successfully!")
