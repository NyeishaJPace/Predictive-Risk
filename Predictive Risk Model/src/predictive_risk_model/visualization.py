"""Visualization helpers for EDA, predictions, and anomaly inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="crest")


def create_eda_visuals(df: pd.DataFrame, figure_dir: Path) -> None:
    """Create histogram and correlation plots for exploratory analysis."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    numeric_columns = [
        "age",
        "bmi",
        "systolic_bp",
        "glucose",
        "cholesterol",
        "inflammation_score",
    ]

    df[numeric_columns].hist(figsize=(12, 8), bins=18, edgecolor="black")
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(figure_dir / "histograms.png", dpi=200)
    plt.close()

    corr_df = df[numeric_columns + ["prior_admissions", "hospitalization_days", "high_risk_outcome"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figure_dir / "correlation_heatmap.png", dpi=200)
    plt.close()


def create_prediction_visuals(
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Plot model outputs for interpretability."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=predictions,
        x="predicted_probability",
        hue="high_risk_outcome",
        bins=18,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Predicted Risk Probability by Actual Outcome")
    plt.xlabel("Predicted probability of high risk")
    plt.tight_layout()
    plt.savefig(figure_dir / "prediction_probability_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(12).sort_values("importance")
    sns.barplot(data=top_features, x="importance", y="feature", color="#2f7e79")
    plt.title("Top Model Features")
    plt.xlabel("Feature importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figure_dir / "feature_importance.png", dpi=200)
    plt.close()


def create_anomaly_visuals(anomaly_df: pd.DataFrame, figure_dir: Path) -> None:
    """Plot anomaly patterns to help inspect unusual records."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=anomaly_df,
        x="glucose",
        y="inflammation_score",
        hue="is_anomaly",
        style="high_risk_outcome",
        palette={0: "#4c78a8", 1: "#d62728"},
        alpha=0.75,
    )
    plt.title("Anomaly Detection Across Glucose and Inflammation")
    plt.xlabel("Glucose")
    plt.ylabel("Inflammation score")
    plt.tight_layout()
    plt.savefig(figure_dir / "anomaly_scatter.png", dpi=200)
    plt.close()

