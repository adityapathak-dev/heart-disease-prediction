
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier


# Load the dataset
df = pd.read_csv("heart_disease_uci.csv")
df = df.drop(columns=["id"])

x = df.drop(columns=["num", "dataset"])
y = (df["num"] > 0).astype(int)

# Define feature columns
num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Preprocess numerical and categorical data
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Split data before model selection
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Tune XGBoost without PCA
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])

params = {
    "model__n_estimators": [100, 200, 300, 500, 700],
    "model__max_depth": [2, 3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__gamma": [0, 0.1, 0.3, 0.5],
    "model__reg_lambda": [1, 5, 10, 20]
}

search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=params,
    n_iter=40,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTuning XGBoost...")
search.fit(x_train, y_train)

best_xgb = search.best_estimator_

print("\nBest XGBoost parameters:")
print(search.best_params_)
print("Best CV Macro F1:", round(search.best_score_, 4))


# Build comparison models without PCA
models = {
    "Tuned XGBoost": best_xgb,
    "SVM": Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=501,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "Extra Trees": Pipeline([
        ("preprocessor", preprocessor),
        ("model", ExtraTreesClassifier(
            n_estimators=501,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

results = []

print("\nEvaluating models on the test set...")

for name, model in models.items():
    if name != "Tuned XGBoost":
        model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    auc = roc_auc_score(y_test, probabilities)

    results.append([name, accuracy, macro_f1, auc])

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print(classification_report(
        y_test, predictions, target_names=["No disease", "Disease"]
    ))

# Compare test results
results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Macro F1", "ROC-AUC"]
).sort_values("Macro F1", ascending=False)

print("\nFINAL MODEL COMPARISON:")
print(results.to_string(index=False))

# Save the tuned XGBoost pipeline
joblib.dump(models["Random Forest"], "heart_disease_xgboost_model.pkl")
print("\nSaved tuned XGBoost model to heart_disease_xgboost_model.pkl")