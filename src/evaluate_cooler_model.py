
from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT = Path(__file__).resolve().parent.parent

X = pd.read_csv(
    PROJECT / "data" / "processed" / "sensor_features.csv"
)

profile = pd.read_csv(
    PROJECT / "data" / "raw" / "profile.txt",
    sep=r"\s+",
    header=None,
    engine="python"
)

if len(X) != len(profile):
    raise ValueError("Feature and label row counts do not match!")

y = profile.iloc[:, 0]

# Keep identical complete operating profiles together
groups = profile.astype(str).agg("_".join, axis=1)

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

print("Training cycles:", len(X_train))
print("Testing cycles:", len(X_test))
print("Operating profiles in test:",
      groups.iloc[test_idx].nunique())

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining with grouped split...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nGrouped test accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
