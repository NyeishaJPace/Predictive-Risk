# Predictive Risk Model

This project builds a modular Python data science workflow around simulated public health data. It loads multiple source files in CSV and JSON format, cleans and merges them into one analytical dataset, performs exploratory data analysis, trains a binary classification model to predict patient risk, detects anomalous records, and saves reusable outputs to disk.

## Project Structure

- `data/raw/`: generated source datasets in CSV and JSON format
- `data/processed/`: cleaned data, predictions, anomaly scores, and feature importances
- `models/`: trained model and evaluation metrics
- `reports/figures/`: EDA, prediction, and anomaly plots
- `src/predictive_risk_model/`: modular pipeline code
- `run_pipeline.py`: entry point for the full workflow

## Methodology

1. **Data simulation and loading**
   The project generates three deterministic public-health datasets:
   - patient demographics (`patient_demographics.csv`)
   - clinical measurements (`clinical_measurements.csv`)
   - social and follow-up factors (`social_factors.json`)

2. **Cleaning and merging**
   - imputes missing numeric values with medians
   - fills missing categorical values with common values
   - engineers interpretable risk-oriented features such as blood-pressure flags and adherence gaps
   - merges all records on `patient_id`

3. **Exploratory data analysis**
   - histograms summarize feature distributions
   - a correlation heatmap highlights relationships with the binary outcome

4. **Predictive modeling**
   - uses a `RandomForestClassifier`
   - preprocesses mixed numeric and categorical columns with a scikit-learn pipeline
   - evaluates performance with accuracy, precision, recall, F1-score, and ROC-AUC

5. **Anomaly detection**
   - uses `IsolationForest` on numeric features
   - assigns anomaly scores and flags unusual patient records

6. **Interpretability**
   - plots predicted-risk distributions by actual outcome
   - ranks top model features by importance
   - visualizes anomaly clusters across clinical measures

## How to Run

From the project directory:

```bash
PYTHONPATH=src python run_pipeline.py
```

## Saved Outputs

- `data/processed/final_modeling_dataset.csv`
- `data/processed/test_predictions.csv`
- `data/processed/anomaly_scored_dataset.csv`
- `data/processed/feature_importance.csv`
- `models/risk_model.joblib`
- `models/metrics.json`
- `reports/run_summary.json`
- `reports/figures/*.png`

## Results

The exact evaluation metrics are written to `models/metrics.json` and summarized in `reports/run_summary.json` each time the pipeline runs. Because the data is deterministic, the project is reproducible across runs in the same environment.
