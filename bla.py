"""
Bike-Sharing Drift Monitoring — Evidently Exam
================================================
Single-run script that produces all required reports:
  01_model_validation.html
  02_production_january.html
  03_week1_drift.html
  04_week2_drift.html
  05_week3_drift.html
  06_target_drift_worst_week.html
  07_data_drift_week3_numerical.html
"""

import datetime
import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn import ensemble, model_selection

from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
REPORTS_DIR = Path("reports")

TARGET = "cnt"
PREDICTION = "prediction"
NUMERICAL_FEATURES = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday"]

# Week slices for February monitoring (week 1 starts end of January per exam spec)
WEEKS = {
    "week1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_data() -> pd.DataFrame:
    """Download and parse the raw bike-sharing CSV from UCI."""
    content = requests.get(URL, verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw = pd.read_csv(
            arc.open("hour.csv"),
            header=0,
            sep=",",
            parse_dates=["dteday"],
        )
    return raw


def process_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Set a proper datetime index from date + hour columns."""
    raw.index = raw.apply(
        lambda row: datetime.datetime.combine(
            row.dteday.date(), datetime.time(row.hr)
        ),
        axis=1,
    )
    return raw


# ---------------------------------------------------------------------------
# Evidently helpers
# ---------------------------------------------------------------------------

def make_column_mapping(
    target: str,
    prediction: str,
    numerical_features: list,
    categorical_features: list,
) -> ColumnMapping:
    """Build a reusable ColumnMapping object."""
    cm = ColumnMapping()
    cm.target = target
    cm.prediction = prediction
    cm.numerical_features = numerical_features
    cm.categorical_features = categorical_features
    return cm


def run_report(metrics: list, reference: pd.DataFrame, current: pd.DataFrame,
               column_mapping: ColumnMapping) -> Report:
    """Run an Evidently Report and return it."""
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    return report


def save_report(report: Report, name: str) -> None:
    """Save report as HTML to the reports directory."""
    path = REPORTS_DIR / f"{name}.html"
    report.save_html(str(path))
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Load data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading data...")
    raw_data = process_data(fetch_data())

    # Reference = January 2011 (training window)
    reference_jan11 = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"].copy()
    # Current window covers the Feb monitoring period (+ end of Jan for week 1)
    current_feb11 = raw_data.loc["2011-01-29 00:00:00":"2011-02-28 23:00:00"].copy()

    # ------------------------------------------------------------------
    # STEP 1 — Train/test split on January data
    # ------------------------------------------------------------------
    print("\n[2/7] Step 1 — Train/test split & initial model training...")
    X = reference_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = reference_jan11[TARGET]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
    regressor.fit(X_train, y_train)

    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)

    # ------------------------------------------------------------------
    # STEP 2 — Model validation report (train as reference, test as current)
    # ------------------------------------------------------------------
    print("\n[3/7] Step 2 — Model validation report...")

    # Validation uses renamed 'target' column because cnt is split off into y
    X_train_val = X_train.copy()
    X_train_val["target"] = y_train
    X_train_val["prediction"] = preds_train

    X_test_val = X_test.copy()
    X_test_val["target"] = y_test
    X_test_val["prediction"] = preds_test

    cm_validation = make_column_mapping(
        target="target",
        prediction="prediction",
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    report_validation = run_report(
        metrics=[RegressionPreset()],
        reference=X_train_val.sort_index(),
        current=X_test_val.sort_index(),
        column_mapping=cm_validation,
    )
    save_report(report_validation, "01_model_validation")

    # ------------------------------------------------------------------
    # STEP 3 — Production model: retrain on full January dataset
    # ------------------------------------------------------------------
    print("\n[4/7] Step 3 — Production model (full January)...")

    regressor.fit(
        reference_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES],
        reference_jan11[TARGET],
    )

    # Add predictions to the reference dataset for downstream reports
    reference_jan11[PREDICTION] = regressor.predict(
        reference_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )

    # Column mapping for production reports uses original 'cnt' column
    cm_production = make_column_mapping(
        target=TARGET,
        prediction=PREDICTION,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # Production model self-check: reference=None → single-dataset report
    report_production = run_report(
        metrics=[RegressionPreset()],
        reference=None,
        current=reference_jan11,
        column_mapping=cm_production,
    )
    save_report(report_production, "02_production_january")

    # ------------------------------------------------------------------
    # STEP 4 — Weekly monitoring reports (weeks 1, 2, 3)
    # ------------------------------------------------------------------
    print("\n[5/7] Step 4 — Weekly drift reports...")

    week_rmse = {}  # collect RMSE per week to identify the worst week

    for week_name, (start, end) in WEEKS.items():
        week_data = current_feb11.loc[start:end].copy()

        # Generate predictions for the current week
        week_data[PREDICTION] = regressor.predict(
            week_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )

        report_week = run_report(
            metrics=[RegressionPreset()],
            reference=reference_jan11,
            current=week_data,
            column_mapping=cm_production,
        )

        report_index = {"week1": "03", "week2": "04", "week3": "05"}[week_name]
        save_report(report_week, f"{report_index}_{week_name}_drift")

        # Compute RMSE to rank weeks (used for step 5)
        rmse = np.sqrt(
            np.mean((week_data[TARGET].values - week_data[PREDICTION].values) ** 2)
        )
        week_rmse[week_name] = rmse
        print(f"    {week_name} RMSE: {rmse:.2f}")

    # Identify the worst week by highest RMSE
    worst_week_name = max(week_rmse, key=week_rmse.get)
    worst_start, worst_end = WEEKS[worst_week_name]
    print(f"\n  Worst week: {worst_week_name} (RMSE={week_rmse[worst_week_name]:.2f})")

    # ------------------------------------------------------------------
    # STEP 5 — Target drift report on the worst week
    # ------------------------------------------------------------------
    print("\n[6/7] Step 5 — Target drift report (worst week)...")

    worst_week_data = current_feb11.loc[worst_start:worst_end].copy()
    worst_week_data[PREDICTION] = regressor.predict(
        worst_week_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )

    # Categorical features excluded for target drift analysis per exam template
    cm_target_drift = make_column_mapping(
        target=TARGET,
        prediction=PREDICTION,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=[],
    )

    report_target_drift = run_report(
        metrics=[TargetDriftPreset()],
        reference=reference_jan11,
        current=worst_week_data,
        column_mapping=cm_target_drift,
    )
    save_report(report_target_drift, f"06_target_drift_{worst_week_name}")

    # ------------------------------------------------------------------
    # STEP 6 — Data drift report: week 3, numerical features only
    # ------------------------------------------------------------------
    print("\n[7/7] Step 6 — Data drift report (week 3, numerical only)...")
_
    week3_start, week3_end = WEEKS["week3"]
    week3_data = current_feb11.loc[week3_start:week3_end].copy()
    week3_data[PREDICTION] = regressor.predict(
        week3_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )

    # Restrict to numerical features only as required by step 6
    cm_data_drift = make_column_mapping(
        target=TARGET,
        prediction=PREDICTION,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=[],  # excluded — numerical only per exam spec
    )

    report_data_drift = run_report(
        metrics=[DataDriftPreset()],
        reference=reference_jan11,
        current=week3_data,
        column_mapping=cm_data_drift,
    )
    save_report(report_data_drift, "07_data_drift_week3_numerical")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All reports generated successfully.")
    print(f"Output directory: {REPORTS_DIR.resolve()}")
    print("=" * 60)
    print("\nWeek RMSE summary:")
    for week, rmse in week_rmse.items():
        marker = " <-- worst" if week == worst_week_name else ""
        print(f"  {week}: {rmse:.2f}{marker}")
    print()


if __name__ == "__main__":
    main()