import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json
from pathlib import Path

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.report import Report
from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# Variables
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

TARGET = 'cnt'
PREDICTION = 'prediction'
NUMERICAL_FEATURES = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CATEGORICAL_FEATURES = ['season', 'holiday', 'workingday']

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEEKS = {
    "week1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}



# Helper Funcs

def _fetch_data() -> pd.DataFrame:
    content = requests.get(URL, verify=False).content

    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(
            arc.open("hour.csv"),
            header=0,
            sep=',',
            parse_dates=['dteday']
    ) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row.dteday.date(),
            datetime.time(row.hr)
            ),
            axis=1
        )
    return raw_data

def make_column_mapping(
        target: str,
        prediction: str,
        numerical_features: list[str],
        categorical_features: list[str]
):
    cm = ColumnMapping()
    cm.target = target
    cm.prediction = prediction
    cm.numerical_features = numerical_features
    cm.categorical_features = categorical_features
    return cm

def save_report(report: Report, name: str) -> None:
    """Saves the html Reports"""
    out = REPORTS_DIR / f"{name}.html"
    report.save_html(str(out))

# ------------------------------------------------------------------------
# load data - Step 1 
# ------------------------------------------------------------------------

raw_data = _process_data(_fetch_data())

ref_jan11 = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
cur_feb11 = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']
print("\nStep 1: Data loaded and processed successfully!")


# misc

# Reference and current data splitpip show evidently
X = ref_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = ref_jan11[TARGET]



def main():

    # ------------------------------------------------------------------------
    # test Run on Jan - Step 2 
    # -----------------------------------------------------------------------

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # Model training
    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(X_train, y_train)

    # Predictions
    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)


    # Add actual target and prediction columns to the training data for later performance analysis
    X_train_df = X_train.copy()  
    X_train_df['target'] = y_train    # target as y_train (70 percent of data)
    X_train_df['prediction'] = preds_train

    X_test_df = X_test.copy()
    X_test_df['target'] = y_test
    X_test_df['prediction'] = preds_test

    # # Initialize the column mapping object, for eval purpose for evidently 
    cm = make_column_mapping(
        target='target',
        prediction='prediction',
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    # Use Regression evaluation metric for test and whole Dataste STep 2&3
    regression_performance_report = Report(metrics=[
        RegressionPreset()
    ])
    regression_performance_report.run(
        reference_data=X_train_df.sort_index(), 
        current_data=X_test_df.sort_index(),
        column_mapping=cm
    )

    save_report(regression_performance_report, "01_jan_test")
    print("Step 2: Report saved successfully!")



    # ------------------------------------------------------------------------
    # test Run on Jan - Step 3
    # ------------------------------------------------------------------------



    regressor.fit(
        ref_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES],
        ref_jan11[TARGET]
    )

    # prediction col gets added. Train data whole month!
    ref_jan11['prediction'] = regressor.predict(                
        ref_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )

    cm = make_column_mapping(
        target=TARGET,
        prediction=PREDICTION,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    # Metric on DATA jan COMPLETE
    # regression_performance_report from above
    regression_performance_report.run(
        reference_data=None,                   # were tested in step 2
        current_data=ref_jan11.sort_index(),
        column_mapping=cm
    )
    save_report(regression_performance_report, "02_jan_prod")
    print(f"Step 3: Report for prod saved successfully!")


    # ------------------------------------------------------------------
    # Weekly monitoring Step 4: reports (weeks 1, 2, 3)
    # ------------------------------------------------------------------
    week_rmse = {} # saving rmse states per week
    # Comparing predict full train data from jan11, applied to feb11, per week data
    for week_name, (start, end) in WEEKS.items():
        week_data = cur_feb11.loc[start:end].copy()             # start end fix boundary dates

        week_data[PREDICTION] = regressor.predict(
            week_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )

        cm = make_column_mapping(
            target=TARGET,
            prediction=PREDICTION,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES
        )

        regression_performance_report = Report(metrics=[RegressionPreset()])
        regression_performance_report.run(
            reference_data=ref_jan11,
            current_data=week_data,
            column_mapping=cm
        )


        # Saving
        report_index = {
            "week1": "03",
            "week2": "04",
            "week3": "05"
        }[week_name]

        save_report(regression_performance_report, f"{report_index}_drift_for_{week_name}")
        print(f"Step 4: Reports for {week_name} saved successfully!")

        rmse = np.sqrt(
            np.mean((week_data[TARGET].values - week_data[PREDICTION].values)**2)
        )
        week_rmse[week_name] = rmse
        print(f"RMSE for {week_name}: {rmse:.2f}")

    # call the states
    rmse_max = max(week_rmse, key=week_rmse.get)
    # gets dates back from specific week
    worst_start, worst_end = WEEKS[rmse_max]
    print(f"\n  Worst week: {rmse_max} (RMSE={week_rmse[rmse_max]:.2f})")



    # ------------------------------------------------------------------
    # Target drift report on the worst week
    # ------------------------------------------------------------------

    worst_week_data = cur_feb11.loc[worst_start:worst_end].copy()
    worst_week_data[PREDICTION] = regressor.predict(
        worst_week_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )

    cm = make_column_mapping(
        target=TARGET,
        prediction=PREDICTION,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    Target_Drift_preset_report = Report(metrics=[
    TargetDriftPreset()
    ])

    Target_Drift_preset_report.run(
        reference_data=ref_jan11.sort_index(),
        current_data=worst_week_data.sort_index(),
        column_mapping=cm
    )
    save_report(Target_Drift_preset_report, f"06_target_drift_{rmse_max}")



    # ------------------------------------------------------------------
    # Target drift report on the worst week
    # ------------------------------------------------------------------

    # week 3 same like worst week

    week3_start, week3_end = WEEKS["week3"]
    # week3_data = cur_feb11[week3_start:week3_end]
    week3_data = cur_feb11.loc[week3_start:week3_end].copy()

    week3_data[PREDICTION] = regressor.predict(
        week3_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )

    cm = make_column_mapping(
    target=TARGET,
    prediction=PREDICTION,
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=[]
    )


    DataDriftPreset_Report = Report(
        metrics = [DataDriftPreset()]
    )
    DataDriftPreset_Report.run(
        reference_data=ref_jan11.sort_index(),
        current_data=week3_data.sort_index(),
        column_mapping=cm
    )
    save_report(DataDriftPreset_Report, f"07_data_drift_{rmse_max}")



if __name__ == "__main__":
    main()
