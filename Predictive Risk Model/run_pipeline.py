"""Run the full predictive risk modeling workflow."""

from __future__ import annotations

import json
from pathlib import Path

from predictive_risk_model.modeling import (
    detect_anomalies,
    get_feature_importance,
    save_artifacts,
    train_predictive_model,
)
from predictive_risk_model.pipeline import clean_and_merge, generate_sample_datasets, get_feature_columns, load_raw_datasets
from predictive_risk_model.visualization import create_anomaly_visuals, create_eda_visuals, create_prediction_visuals


def main() -> None:
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    figure_dir = project_root / "reports" / "figures"

    # Create deterministic source files so the project is reproducible out of the box.
    generate_sample_datasets(raw_dir)

    demographics, clinical, social = load_raw_datasets(raw_dir)
    cleaned_df = clean_and_merge(demographics, clinical, social)
    numeric_features, categorical_features = get_feature_columns(cleaned_df)

    model, metrics, predictions = train_predictive_model(cleaned_df, numeric_features, categorical_features)
    anomaly_df = detect_anomalies(cleaned_df, numeric_features)
    feature_importance = get_feature_importance(model)

    create_eda_visuals(cleaned_df, figure_dir)
    create_prediction_visuals(predictions, feature_importance, figure_dir)
    create_anomaly_visuals(anomaly_df, figure_dir)

    save_artifacts(
        model=model,
        metrics=metrics,
        cleaned_df=cleaned_df,
        predictions=predictions,
        anomaly_df=anomaly_df,
        feature_importance=feature_importance,
        output_dirs={"processed": processed_dir, "models": models_dir},
    )

    summary = {
        "rows_in_final_dataset": int(cleaned_df.shape[0]),
        "columns_in_final_dataset": int(cleaned_df.shape[1]),
        "anomalies_flagged": int(anomaly_df["is_anomaly"].sum()),
        "metrics": metrics,
    }
    (project_root / "reports" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

