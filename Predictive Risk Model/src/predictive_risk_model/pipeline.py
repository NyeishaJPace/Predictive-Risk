"""Data generation, loading, and feature engineering utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_datasets(data_dir: Path, seed: int = 42, n_rows: int = 500) -> None:
    """Create deterministic raw datasets if they do not already exist."""
    data_dir.mkdir(parents=True, exist_ok=True)

    demographics_path = data_dir / "patient_demographics.csv"
    clinical_path = data_dir / "clinical_measurements.csv"
    social_path = data_dir / "social_factors.json"

    rng = np.random.default_rng(seed)
    patient_ids = np.arange(1000, 1000 + n_rows)

    demographics = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": rng.integers(20, 90, size=n_rows),
            "gender": rng.choice(["Female", "Male"], size=n_rows),
            "region": rng.choice(["North", "South", "East", "West"], size=n_rows),
            "smoker": rng.choice(["yes", "no"], size=n_rows, p=[0.28, 0.72]),
            "prior_admissions": rng.poisson(lam=1.5, size=n_rows),
        }
    )

    bmi = rng.normal(29, 5.8, size=n_rows).clip(17, 48)
    systolic_bp = rng.normal(132, 18, size=n_rows).clip(90, 210)
    diastolic_bp = rng.normal(82, 11, size=n_rows).clip(55, 130)
    glucose = rng.normal(112, 28, size=n_rows).clip(65, 280)
    cholesterol = rng.normal(204, 34, size=n_rows).clip(120, 360)
    inflammation_score = rng.normal(4.3, 1.7, size=n_rows).clip(0.2, 10.0)
    hospitalization_days = rng.poisson(lam=4.2, size=n_rows) + 1
    medication_adherence = rng.uniform(0.45, 0.99, size=n_rows).round(2)

    risk_signal = (
        -0.8
        + 0.95 * ((demographics["age"] - 52) / 16)
        + 0.55 * ((bmi - 28) / 5.5)
        + 0.75 * ((systolic_bp - 130) / 18)
        + 0.85 * ((glucose - 110) / 24)
        + 0.45 * ((cholesterol - 200) / 32)
        + 0.70 * demographics["prior_admissions"]
        + 1.15 * (demographics["smoker"] == "yes").astype(int)
        + 1.10 * (medication_adherence < 0.7).astype(int)
        + 0.90 * ((inflammation_score - 4.2) / 1.6)
        + rng.normal(0, 0.35, size=n_rows)
    )
    risk_probability = 1 / (1 + np.exp(-risk_signal))
    high_risk_outcome = rng.binomial(1, risk_probability)

    clinical = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "bmi": bmi.round(1),
            "systolic_bp": systolic_bp.round(0).astype(int),
            "diastolic_bp": diastolic_bp.round(0).astype(int),
            "glucose": glucose.round(1),
            "cholesterol": cholesterol.round(1),
            "inflammation_score": inflammation_score.round(2),
            "hospitalization_days": hospitalization_days,
            "medication_adherence": medication_adherence,
            "high_risk_outcome": high_risk_outcome,
        }
    )

    insurance = rng.choice(
        ["Private", "Medicaid", "Medicare", "Uninsured"],
        size=n_rows,
        p=[0.46, 0.2, 0.25, 0.09],
    )
    telehealth = rng.choice(["completed", "missed"], size=n_rows, p=[0.74, 0.26])
    exercise_level = rng.choice(["low", "moderate", "high"], size=n_rows, p=[0.34, 0.48, 0.18])

    social_records = []
    for idx, patient_id in enumerate(patient_ids):
        social_records.append(
            {
                "patient_id": int(patient_id),
                "insurance_type": insurance[idx],
                "telehealth_follow_up": telehealth[idx],
                "exercise_level": exercise_level[idx],
                "vaccinated": bool(rng.choice([True, False], p=[0.78, 0.22])),
                "household_size": int(rng.integers(1, 7)),
                "community_risk_index": float(np.round(rng.uniform(0.1, 0.95), 2)),
            }
        )

    # Introduce a small amount of missingness so the cleaning step has work to do.
    for frame, columns in [
        (demographics, ["age", "smoker"]),
        (clinical, ["bmi", "glucose", "cholesterol", "medication_adherence"]),
    ]:
        for column in columns:
            missing_indices = rng.choice(frame.index, size=max(5, n_rows // 40), replace=False)
            frame.loc[missing_indices, column] = np.nan

    demographics.to_csv(demographics_path, index=False)
    clinical.to_csv(clinical_path, index=False)
    social_path.write_text(json.dumps(social_records, indent=2), encoding="utf-8")


def load_raw_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw CSV and JSON source files."""
    demographics = pd.read_csv(data_dir / "patient_demographics.csv")
    clinical = pd.read_csv(data_dir / "clinical_measurements.csv")
    social = pd.read_json(data_dir / "social_factors.json")
    return demographics, clinical, social


def clean_and_merge(
    demographics: pd.DataFrame,
    clinical: pd.DataFrame,
    social: pd.DataFrame,
) -> pd.DataFrame:
    """Clean each dataset and merge them into a single modeling frame."""
    demographics = demographics.copy()
    clinical = clinical.copy()
    social = social.copy()

    for frame in (demographics, clinical, social):
        frame["patient_id"] = frame["patient_id"].astype(int)

    numeric_fill_columns = ["age", "bmi", "glucose", "cholesterol", "medication_adherence"]
    for column in numeric_fill_columns:
        if column in demographics:
            demographics[column] = demographics[column].fillna(demographics[column].median())
        if column in clinical:
            clinical[column] = clinical[column].fillna(clinical[column].median())

    demographics["smoker"] = demographics["smoker"].fillna(demographics["smoker"].mode().iloc[0])
    social["telehealth_follow_up"] = social["telehealth_follow_up"].fillna("completed")
    social["exercise_level"] = social["exercise_level"].fillna("moderate")

    merged = demographics.merge(clinical, on="patient_id", how="inner").merge(social, on="patient_id", how="inner")

    merged["smoker_flag"] = (merged["smoker"].str.lower() == "yes").astype(int)
    merged["vaccinated_flag"] = merged["vaccinated"].astype(int)
    merged["telehealth_missed_flag"] = (merged["telehealth_follow_up"] == "missed").astype(int)
    merged["high_bp_flag"] = ((merged["systolic_bp"] >= 140) | (merged["diastolic_bp"] >= 90)).astype(int)
    merged["glucose_risk_flag"] = (merged["glucose"] >= 126).astype(int)
    merged["adherence_gap"] = (1 - merged["medication_adherence"]).round(2)
    merged["bmi_risk_group"] = pd.cut(
        merged["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "healthy", "overweight", "obese"],
        include_lowest=True,
    )

    return merged.sort_values("patient_id").reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature column names used in the model."""
    numeric_features = [
        "age",
        "prior_admissions",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "glucose",
        "cholesterol",
        "inflammation_score",
        "hospitalization_days",
        "medication_adherence",
        "household_size",
        "community_risk_index",
        "smoker_flag",
        "vaccinated_flag",
        "telehealth_missed_flag",
        "high_bp_flag",
        "glucose_risk_flag",
        "adherence_gap",
    ]
    categorical_features = ["gender", "region", "insurance_type", "exercise_level", "bmi_risk_group"]
    return numeric_features, categorical_features
