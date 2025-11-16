import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from . import data_preprocessing
import features.feature_engineering as feature_engineering

def train_test_split(df):
    train_set = df[df['datetime'] < '2023-01-01'].reset_index(drop=True)
    test_set  = df[df['datetime'] >= '2023-01-01'].reset_index(drop=True)
    return train_set, test_set

before_model_pipeline = Pipeline(steps=[
    ('remove_low_variance', data_preprocessing.remove_low_variance_pipeline),
    ('data_transformation', data_preprocessing.DataTransformer(categorical_features=['conditions'])),
    ('feature_engineering', feature_engineering.feature_engineering_class())
]
)