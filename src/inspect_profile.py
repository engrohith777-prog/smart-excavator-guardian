
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"

profile_file = raw_folder / "profile.txt"

profile = pd.read_csv(
    profile_file,
    sep=r"\s+",
    header=None,
    engine="python"
)

print("Profile shape:", profile.shape)
print("\nColumn names:")
print([
    "cooler_condition",
    "valve_condition",
    "internal_pump_leakage",
    "accumulator_pressure",
    "stable_flag"
])

print("\nFirst 10 profile rows:")
print(profile.head(10).to_string(index=False))

features = pd.read_csv(
    project_folder / "data" / "processed" / "sensor_features.csv"
)

print("\nFeature rows:", len(features))
print("Profile rows:", len(profile))
print("\nProfile value counts:")
for column in profile.columns:
    print(f"\nColumn {column}:")
    print(profile[column].value_counts().sort_index())
