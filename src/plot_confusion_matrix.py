
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT / "reports" / "figures" / "cooler_confusion_matrix.csv"
OUTPUT_FILE = PROJECT / "reports" / "figures" / "cooler_confusion_matrix.png"

matrix_df = pd.read_csv(DATA_FILE, index_col=0)
matrix = matrix_df.to_numpy()
labels = ["3", "20", "100"]

fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Cooler Condition Classification", pad=15)
ax.set_xlabel("Predicted condition")
ax.set_ylabel("Actual condition")

fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Image saved to: {OUTPUT_FILE}")
