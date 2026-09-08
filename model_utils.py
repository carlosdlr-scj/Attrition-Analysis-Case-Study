from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import randint, uniform, loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "HR_capstone_dataset.csv"

NUMERIC_FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
]
CATEGORICAL_FEATURES = ["Department", "salary"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def prep(num, cat):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )


def _metric_row(name, estimator, X_test, y_test, random_best=None, grid_best=None):
    pred = estimator.predict(X_test)
    prob = estimator.predict_proba(X_test)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }
    if random_best is not None:
        row["RandomizedSearchCV Best Params"] = str(random_best)
    if grid_best is not None:
        row["GridSearchCV Best Params"] = str(grid_best)
    return row


def _grid_from_random_best(model_name, best_params):
    """Focused refinement grid around RandomizedSearchCV's winner."""
    if model_name == "Logistic Regression":
        c = float(best_params["model__C"])
        return {"model__C": [round(c * 0.85, 4), round(c, 4)], "model__class_weight": ["balanced"]}
    if model_name == "Random Forest":
        n = int(best_params["model__n_estimators"])
        d = best_params["model__max_depth"]
        depths = [d] if d is None else sorted(set([max(3, int(d) - 1), int(d)]))
        return {
            "model__n_estimators": [max(100, n - 50), n],
            "model__max_depth": depths,
            "model__min_samples_split": [best_params["model__min_samples_split"]],
            "model__min_samples_leaf": [best_params["model__min_samples_leaf"]],
            "model__max_features": [best_params["model__max_features"]],
            "model__class_weight": ["balanced"],
        }
    n = int(best_params["model__n_estimators"])
    lr = float(best_params["model__learning_rate"])
    d = int(best_params["model__max_depth"])
    return {
        "model__n_estimators": [max(50, n - 50), n],
        "model__learning_rate": [round(lr * 0.85, 4), round(lr, 4)],
        "model__max_depth": [max(1, d - 1), d],
        "model__min_samples_split": [best_params["model__min_samples_split"]],
        "model__min_samples_leaf": [best_params["model__min_samples_leaf"]],
        "model__subsample": [best_params["model__subsample"]],
    }


@st.cache_resource

def train_models():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df.left
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_specs = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "random_params": {
                "model__C": loguniform(0.01, 100),
                "model__class_weight": ["balanced"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
            "random_params": {
                "model__n_estimators": randint(100, 201),
                "model__max_depth": [None, 5, 8, 10, 12, 15, 20, 25, 30],
                "model__min_samples_split": randint(2, 11),
                "model__min_samples_leaf": randint(1, 6),
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": ["balanced"],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "random_params": {
                "model__n_estimators": randint(50, 151),
                "model__learning_rate": uniform(0.03, 0.15),
                "model__max_depth": randint(2, 6),
                "model__min_samples_split": randint(2, 11),
                "model__min_samples_leaf": randint(1, 6),
                "model__subsample": uniform(0.7, 0.3),
            },
        },
    }

    fitted = {}
    rows = []

    for name, spec in model_specs.items():
        pipe = Pipeline(
            [
                ("prep", prep(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                ("model", spec["estimator"]),
            ]
        )

        randomized = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec["random_params"],
            n_iter=15,
            scoring="recall",
            cv=3,
            random_state=42,
            n_jobs=3,
            refit=True,
        )
        randomized.fit(X_train, y_train)

        refinement_grid = _grid_from_random_best(name, randomized.best_params_)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=refinement_grid,
            scoring="recall",
            cv=3,
            n_jobs=3,
            refit=True,
        )
        grid.fit(X_train, y_train)

        fitted[name] = grid.best_estimator_
        rows.append(
            _metric_row(
                name,
                grid.best_estimator_,
                X_test,
                y_test,
                randomized.best_params_,
                grid.best_params_,
            )
        )

    results = pd.DataFrame(rows).sort_values("Recall", ascending=False).reset_index(drop=True)
    return df, results, fitted, results.loc[0, "Model"]
