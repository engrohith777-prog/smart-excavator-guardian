
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
raw_data_folder = project_folder / "data" / "raw"

print("SMART EXCAVATOR GUARDIAN")
print("-----------------------")
print("Project folder:", project_folder)
print("Raw data folder:", raw_data_folder)
print("Does raw data folder exist?", raw_data_folder.exists())

files = list(raw_data_folder.iterdir())
print("Files currently in raw data folder:", len(files))

for file in files:
    print("-", file.name)
