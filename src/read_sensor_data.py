
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"

print("HYDRAULIC SENSOR DATA INSPECTION")

files = sorted(raw_folder.glob("*.txt"))
print("TXT files found:", len(files))

for file in files:
    try:
        data = pd.read_csv(file, sep=r"\s+", header=None, nrows=5)
        print(f"\nFile: {file.name}")
        print("Sample shape:", data.shape)
        print(data.head(2).to_string(index=False, header=False))
    except Exception as error:
        print(f"Could not read {file.name}: {error}")

print("\nInspection completed.")
