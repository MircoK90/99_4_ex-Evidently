import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json
import pathlib as path

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.report import Report  # <-- fix
from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


# custom functions
def _fetch_data() -> pd.DataFrame:
    content = requests.get(URL, verify=False).content

    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)), axis=1)
    return raw_data



# load data
raw_data = _process_data(_fetch_data())
print(raw_data.head())

# Feature selection
target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']


# like the purposel of the exam
def column_mapping(
        target: str,
        prediction: str,
        numerical_features: list[str],
        categorical_features: list[str]
):
    column_mapping = ColumnMapping()
    column_mapping.target = target,
    column_mapping.prediction = prediction,
    column_mapping.numerical_features = numerical_features,
    column_mapping.categorical_features = categorical_features,
    return column_mapping


REPORTS_DIR = path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)




def main():
    # Reference and current data splitpip show evidently
    reference_jan11 = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
    current_feb11 = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[numerical_features + categorical_features],
        reference_jan11[target],
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
    # mk gets shaped back with y component and adds preds with another Column"
    X_train['target'] = y_train    # in former file als 'cnt
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data for later performance analysis
    X_test['target'] = y_test
    X_test['prediction'] = preds_test

    # # Initialize the column mapping object, which is evidently used to know how the data is structured. 
    column_mapping = ColumnMapping()

    # # Map the actual target and prediction column names in the dataset for evidently
    # column_mapping.target = 'target'
    # column_mapping.prediction = 'prediction'

    # # Specify which features are numerical and which are categorical for the evidently report
    # column_mapping.numerical_features = numerical_features
    # column_mapping.categorical_features = categorical_features

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the training data as reference and test data as current
    # The data is sorted by index to ensure consistent ordering for the comparison
    regression_performance_report.run(reference_data=X_train.sort_index(), 
                                    current_data=X_test.sort_index(),
                                    column_mapping=column_mapping)


    # Train the production model
    regressor.fit(reference_jan11[numerical_features + categorical_features], reference_jan11[target])

    # # Perform column mapping
    # column_mapping = ColumnMapping()
    # column_mapping.target = target
    # column_mapping.prediction = prediction
    # column_mapping.numerical_features = numerical_features
    # column_mapping.categorical_features = categorical_features

    # Generate predictions for the reference data
    ref_prediction = regressor.predict(reference_jan11[numerical_features + categorical_features])
    reference_jan11['prediction'] = ref_prediction

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the reference data
    regression_performance_report.run(reference_data=None, 
                                    current_data=reference_jan11,
                                    column_mapping=column_mapping)
    


    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(
        reference_data=reference_jan11,
        current_data=current_feb11.loc['2011-02-14 00:00:00':'2011-02-21 23:00:00'],
        column_mapping=column_mapping_drift,
    )
    save_report(prod_reg_report, "02_production_january")

if __name__ == "__main__":
    main()
