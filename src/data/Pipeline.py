import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from . import data_preprocessing_daily
from . import data_preprocessing_hourly
import src.features.feature_engineering_daily as feature_engineering_daily
import src.features.feature_engineering_hourly as feature_engineering_hourly


def train_test_split(df):
    train_size = int(len(df) * 0.8)
    train_set = df[:train_size].reset_index(drop=True)
    test_set  = df[train_size:].reset_index(drop=True)
    return train_set, test_set

before_model_pipeline = Pipeline(steps=[
    ('remove_low_variance', data_preprocessing_daily.remove_low_variance_pipeline),
    ('data_transformation', data_preprocessing_daily.DataTransformer(categorical_features=['conditions'])),
    ('feature_engineering', feature_engineering_daily.HanoiDailyFE())
]
)

before_model_pipeline_hourly = Pipeline(steps=[
    ('remove_low_variance', data_preprocessing_hourly.remove_low_variance_pipeline),
    ('imputation_missing', data_preprocessing_hourly.impute_pipeline),
    ('feature_engineering', feature_engineering_hourly.HanoiHourlyFE())
]
)
