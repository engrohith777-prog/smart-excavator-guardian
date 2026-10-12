
from pathlib import Path
import pandas as pd

# Set project folders
project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"
processed_folder = project_folder / "data" / "processed"

processed_folder.mkdir(parents=True, exist_ok=True)

# Read only actual sensor files
sensor_names = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1", "CE", "CP", "SE"
]

feature_frames = []

for sensor_name in sensor_names:
    file = raw_folder / f"{sensor_name}.txt"

    if not file.exists():
        print(f"Missing file: {sensor_name}.txt")
        continue

    try:
        data = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            encoding="latin1",
            engine="python"
        )

        # Calculate four features for every cycle
        features = pd.DataFrame({
            f"{sensor_name}_mean": data.mean(axis=1),
            f"{sensor_name}_std": data.std(axis=1),
            f"{sensor_name}_min": data.min(axis=1),
            f"{sensor_name}_max": data.max(axis=1),
        })

        feature_frames.append(features)
        print(f"Processed {sensor_name}: {len(data)} cycles")

    except Exception as error:
        print(f"Error reading {sensor_name}.txt: {error}")

if not feature_frames:
    print("No sensor data was processed. Check data/raw.")
    raise SystemExit

# Combine sensor features
combined = pd.concat(feature_frames, axis=1)

# Save the processed features
output_file = processed_folder / "sensor_features.csv"
combined.to_csv(output_file, index=False)

print("\nFeature preparation completed.")
print("Feature table shape:", combined.shape)
print("Saved to:", output_file)
