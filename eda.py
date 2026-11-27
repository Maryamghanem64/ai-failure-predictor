
import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# 2. Dataset size
print("Shape:", df.shape)

# 3. Column names
print("Columns:", df.columns.tolist())

# 4. First five students
print(df[["G1", "G2", "G3"]].head())

# 5. Missing values
print("Missing values:", df.isnull().sum().sum())

# 6. Final results
print("Failed:", (df["G3"] < 10).sum())
print("Passed:", (df["G3"] >= 10).sum())

# 7. Students with low first-period grades
low = df[df["G1"] < 10]
print("Low G1:", len(low))

# 8. Final results for students with low G1
print("Failed:", (low["G3"] < 10).sum())
print("Passed:", (low["G3"] >= 10).sum())

# 9. Students who failed despite G1 >= 10
students = df[(df["G1"] >= 10) & (df["G3"] < 10)]
print("Failed despite G1 >= 10:", len(students))

# 10. Examine five of these students
print(students[
    ["G1", "G3", "studytime", "failures", "schoolsup"]
].head())

# 11. Previous failures among these students
print(students["failures"].value_counts().sort_index())

# 12. Compare with students who passed
passed = df[(df["G1"] >= 10) & (df["G3"] >= 10)]
print("Passed, no previous failures:",
      (passed["failures"] == 0).sum())
print("Passed, with previous failures:",
      (passed["failures"] > 0).sum())

# 13. Duplicate rows
print("Exact duplicate rows:", df.duplicated().sum())

# 14. Proposed model features
features = [
    "G1", "studytime", "failures", "schoolsup",
    "famsup", "paid", "activities", "higher"
]

print(df[features].head())

# 15. Feature data types
print(df[features].dtypes)
from sklearn.model_selection import train_test_split

X = df[
    ["G1", "studytime", "failures", "schoolsup",
     "famsup", "paid", "activities", "higher"]
]

y = (df["G3"] < 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train))
print("Testing:", len(X_test))
categorical_features = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher"
]
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in categorical_features:
    X_train_encoded[col] = X_train[col].map({"yes": 1, "no": 0})
    X_test_encoded[col] = X_test[col].map({"yes": 1, "no": 0})

print(X_train_encoded.head())
print("Training missing values:",
      X_train_encoded.isnull().sum().sum())

print("Testing missing values:",
      X_test_encoded.isnull().sum().sum())
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_encoded, y_train)

print("Model trained successfully!")
y_pred = model.predict(X_test_encoded)

print("Predictions:", y_pred[:10])
print("Actual:     ", y_test.to_numpy()[:10])
print("Predictions:", y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
baseline_pred = (X_test["G1"] < 10).astype(int)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

print("Baseline Accuracy:", baseline_accuracy)
print("Baseline Precision:", baseline_precision)
print("Baseline Recall:", baseline_recall)
print("Baseline F1 Score:", baseline_f1)
y_train_pred = model.predict(X_train_encoded)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train_encoded,
    y_train,
    cv=cv,
    scoring="f1"
)

print("CV Scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())
improved_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

improved_scores = cross_val_score(
    improved_model,
    X_train_encoded,
    y_train,
    cv=cv,
    scoring="f1"
)

print("Original CV F1:", cv_scores.mean())
print("Improved CV F1:", improved_scores.mean())
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 3, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_encoded, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)
baseline_cv_scores = []

for train_idx, val_idx in cv.split(X_train_encoded, y_train):

    X_val = X_train_encoded.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    val_pred = (X_val["G1"] < 10).astype(int)

    score = f1_score(y_val, val_pred)

    baseline_cv_scores.append(score)

print("Baseline CV Scores:", baseline_cv_scores)

print("Baseline Mean CV F1:",
      sum(baseline_cv_scores) / len(baseline_cv_scores))

print("Best Random Forest CV F1:",
      grid_search.best_score_)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

lr_scores = cross_val_score(
    lr_model,
    X_train_encoded,
    y_train,
    cv=cv,
    scoring="f1"
)

print("Logistic Regression CV Scores:", lr_scores)
print("Logistic Regression Mean CV F1:",
      lr_scores.mean())
from sklearn.model_selection import cross_val_predict

lr_probabilities = cross_val_predict(
    lr_model,
    X_train_encoded,
    y_train,
    cv=cv,
    method="predict_proba"
)

risk_scores = lr_probabilities[:, 1]

print("Class Probabilities:")
print(lr_probabilities[:5])

print("Risk Scores:", risk_scores[:5])
ranking = pd.DataFrame({
    "G1": X_train_encoded["G1"],
    "risk_score": risk_scores,
    "actual": y_train
})

ai_detected = 0
baseline_detected = 0

for fold, (_, val_idx) in enumerate(
    cv.split(X_train_encoded, y_train), start=1
):
    students = ranking.iloc[val_idx].sort_index()

    ai_top10 = students.sort_values(
        "risk_score", ascending=False, kind="stable"
    ).head(10)

    baseline_top10 = students.sort_values(
        "G1", ascending=True, kind="stable"
    ).head(10)

    ai_count = ai_top10["actual"].sum()
    baseline_count = baseline_top10["actual"].sum()

    ai_detected += ai_count
    baseline_detected += baseline_count

    print(f"Fold {fold}: AI = {ai_count}, Baseline = {baseline_count}")

print("AI detected:", ai_detected)
print("Baseline detected:", baseline_detected)
from sklearn.inspection import permutation_importance

importance_scores = []

for train_idx, val_idx in cv.split(X_train_encoded, y_train):

    X_fold_train = X_train_encoded.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]

    X_val = X_train_encoded.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    lr_model.fit(X_fold_train, y_fold_train)

    result = permutation_importance(
        lr_model,
        X_val,
        y_val,
        scoring="f1",
        n_repeats=10,
        random_state=42
    )

    importance_scores.append(result.importances_mean)

importance_df = pd.DataFrame(
    importance_scores,
    columns=X_train_encoded.columns
)

print("Feature Importance:")
print(importance_df.mean().sort_values(ascending=False))
selected_features = [
    "G1",
    "activities",
    "failures"
]

X_train_selected = X_train_encoded[selected_features]

selected_scores = cross_val_score(
    lr_model,
    X_train_selected,
    y_train,
    cv=cv,
    scoring="f1"
)

print("Original Logistic Regression:", lr_scores.mean())
print("Reduced Logistic Regression:", selected_scores.mean())
print("Baseline:", sum(baseline_cv_scores) / len(baseline_cv_scores))
reduced_probabilities = cross_val_predict(
    lr_model,
    X_train_selected,
    y_train,
    cv=cv,
    method="predict_proba"
)

ranking["reduced_risk"] = reduced_probabilities[:, 1]

reduced_detected = 0

for fold, (_, val_idx) in enumerate(
    cv.split(X_train_selected, y_train), start=1
):
    students = ranking.iloc[val_idx].sort_index()

    top10 = students.sort_values(
        "reduced_risk",
        ascending=False,
        kind="stable"
    ).head(10)

    count = top10["actual"].sum()
    reduced_detected += count

    print(f"Fold {fold}: Reduced AI = {count}")

print("Original AI:", ai_detected)
print("Reduced AI:", reduced_detected)
print("Baseline:", baseline_detected)
import joblib
from pathlib import Path
from sklearn.base import clone

final_model = clone(lr_model)

final_model.fit(X_train_encoded, y_train)

Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": final_model,
        "features": X_train_encoded.columns.tolist()
    },
    "models/predictai_model.joblib"
)

print("Model saved successfully!")