"""Model training, evaluation, and anomaly detection helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_predictive_model(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str = "high_risk_outcome",
    random_state: int = 42,
):
    """Train a random forest classifier with preprocessing for mixed data types."""
    X = df[numeric_features + categorical_features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    prediction_frame = X_test.copy()
    prediction_frame[target_column] = y_test.values
    prediction_frame["predicted_label"] = y_pred
    prediction_frame["predicted_probability"] = y_proba
    prediction_frame = prediction_frame.reset_index().rename(columns={"index": "row_id"})

    return model, metrics, prediction_frame


def detect_anomalies(df: pd.DataFrame, numeric_features: list[str], random_state: int = 42) -> pd.DataFrame:
    """Use Isolation Forest to flag unusual patient records."""
    anomaly_frame = df.copy()
    detector = IsolationForest(contamination=0.06, random_state=random_state)
    anomaly_subset = anomaly_frame[numeric_features]
    anomaly_frame["anomaly_label"] = detector.fit_predict(anomaly_subset)
    anomaly_frame["anomaly_score"] = detector.decision_function(anomaly_subset)
    anomaly_frame["is_anomaly"] = (anomaly_frame["anomaly_label"] == -1).astype(int)
    return anomaly_frame


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Extract random forest feature importances after preprocessing."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_artifacts(
    model: Pipeline,
    metrics: dict[str, float],
    cleaned_df: pd.DataFrame,
    predictions: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> None:
    """Persist reusable project outputs to disk."""
    output_dirs["processed"].mkdir(parents=True, exist_ok=True)
    output_dirs["models"].mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_dirs["processed"] / "final_modeling_dataset.csv", index=False)
    predictions.to_csv(output_dirs["processed"] / "test_predictions.csv", index=False)
    anomaly_df.to_csv(output_dirs["processed"] / "anomaly_scored_dataset.csv", index=False)
    feature_importance.to_csv(output_dirs["processed"] / "feature_importance.csv", index=False)
    joblib.dump(model, output_dirs["models"] / "risk_model.joblib")
    (output_dirs["models"] / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

