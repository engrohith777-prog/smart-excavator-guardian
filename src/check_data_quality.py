
from pathlib import Path
import numpy as np
import pandas as pd

# Locate project files
PROJECT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT / "data" / "processed" / "sensor_features.csv"
)
PROFILE_FILE = PROJECT / "data" / "raw" / "profile.txt"

# Check required files
if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Feature file missing: {FEATURE_FILE}")

if not PROFILE_FILE.exists():
    raise FileNotFoundError(f"Profile file missing: {PROFILE_FILE}")

# Load data
X = pd.read_csv(FEATURE_FILE)

profile = pd.read_csv(
    PROFILE_FILE,
    sep=r"\s+",
    header=None,
    engine="python"
)

# Check row alignment
if len(X) != len(profile):
    raise ValueError(
        f"Row mismatch: {len(X)} features, "
        f"{len(profile)} profile rows"
    )

# Check sensor data quality
print("DATA QUALITY REPORT")
print("=" * 40)
print("Number of cycles:", len(X))
print("Number of features:", X.shape[1])
print("Missing values:", X.isna().sum().sum())
print("Duplicate feature rows:", X.duplicated().sum())
print("Infinite values:", np.isinf(X.to_numpy()).sum())
print("Profile rows:", len(profile))

print("\nCooler-condition counts:")
print(profile.iloc[:, 0].value_counts().sort_index())

# Examine cooling-related features by condition
print("\nCooling-related sensor features by condition:")

for column in ["CE_mean", "CE_max", "CP_mean", "CP_max"]:
    if column not in X.columns:
        print(f"\n{column}: not found")
        continue

    print(f"\n{column}:")
    print(
        X.groupby(profile.iloc[:, 0])[column]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

print("\nData quality check completed.")


print("\nCOMPARING COOLER CONDITIONS 3 AND 20")

for column in ["CE_mean", "CE_max", "CP_mean", "CP_max"]:
    print(f"\n{column}")

    print(
        X.assign(cooler_condition=profile.iloc[:, 0])
        .groupby("cooler_condition")[column]
        .agg(["mean", "std"])
        .loc[[3, 20]]
        .round(4)
    )

