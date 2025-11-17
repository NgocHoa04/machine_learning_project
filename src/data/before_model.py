import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from . import data_preprocessing
import features.feature_engineering as feature_engineering


def train_test_split(df):
    train_size = int(len(df) * 0.8)
    train_set = df[:train_size].reset_index(drop=True)
    test_set  = df[train_size:].reset_index(drop=True)
    return train_set, test_set

def train_test_split_hourly(df):
    # Đảm bảo cột datetime là kiểu datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Lọc và Đặt 'datetime' làm Index
    train_set = df[df['datetime'] < '2023-01-01'].set_index('datetime')
    test_set  = df[df['datetime'] >= '2023-01-01'].set_index('datetime')
    
    return train_set, test_set

before_model_pipeline = Pipeline(steps=[
    ('remove_low_variance', data_preprocessing.remove_low_variance_pipeline),
    ('data_transformation', data_preprocessing.DataTransformer(categorical_features=['conditions'])),
    ('feature_engineering', feature_engineering.HanoiDailyFE())
]
)

before_model_pipeline_hourly = Pipeline(steps=[
    ('remove_low_variance', data_preprocessing.remove_low_variance_pipeline),
    ('imputation_missing', data_preprocessing.impute_pipeline),
    ('feature_engineering', feature_engineering.HanoiHourlyFE())
]
)
