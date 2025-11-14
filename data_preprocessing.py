import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Remove features with low variance
class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.numeric_cols = numeric_cols
        self.selector.fit(X[numeric_cols])
        # Save retained columns
        self.retained_cols = numeric_cols[self.selector.get_support()]
        # Save dropped columns for logging
        self.dropped_cols = [col for col in numeric_cols if col not in self.retained_cols]
        print("Dropped columns due to zero variance:", self.dropped_cols)
        return self
    
    def transform(self, X):
        cols_to_keep = list(self.retained_cols) + list(X.columns.difference(self.numeric_cols))
        return X[cols_to_keep]
    
# Reomve constant columns and duplicated rows
class ConstantAndDuplicateRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Define constant columns
        self.constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        print("Dropped columns due to constant value:", self.constant_cols)
        return self
    
    def transform(self, X):
        # Remove constant columns
        X_cleaned = X.drop(columns=self.constant_cols, errors="ignore")
        # Remove duplicate rows (no longer printing the count)
        X_cleaned = X_cleaned.drop_duplicates()
        return X_cleaned

# Pipeline to remove low variance and constant features
remove_low_variance_pipeline = Pipeline(steps=[
    ('constant_and_duplicate_remover', ConstantAndDuplicateRemover()),
    ('variance_threshold_selector', VarianceThresholdSelector(threshold=0.0))
    ])


# DATA TRANSFORMATION

def precip_scale_function():
    return FunctionTransformer(func=lambda x: np.log1p(x),
                               validate=True,
                               feature_names_out='one-to-one')

def solarradiation_scale_function():
    return FunctionTransformer(
        lambda x: np.log1p(x)/np.log1p(x.max()), 
        validate=True,
        feature_names_out='one-to-one'
    )

def humidity_scale_function():
    return FunctionTransformer(
        lambda x: x / 100.0, 
        validate=True,
        feature_names_out='one-to-one'
    )

def percentage_scale_function():
    return FunctionTransformer(
        lambda x: x / 100.0, 
        validate=True,
        feature_names_out='one-to-one'
    )

def windspeed_scale_function():
    return FunctionTransformer(
        lambda x: x / 100.0, 
        validate=True,
        feature_names_out='one-to-one'
    )

class DataTransformer:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        self.numeric_transformer = ColumnTransformer(transformers=[
            ('percentage_scaler', percentage_scale_function(), 
             ['precipprob', 'cloudcover', 'precipcover']),
            ('windspeed_scaler', windspeed_scale_function(), 
             ['windspeed']),
            ('precip_scaler', precip_scale_function(), 
             ['precip']),
            ('solarradiation_scaler', solarradiation_scale_function(), 
             ['solarradiation']),
            ('humidity_scaler', humidity_scale_function(), 
             ['humidity'])
        ], remainder='passthrough')
        
        self.categorical_transformer = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')
        
        self.pipeline = Pipeline(steps=[
            ('numeric_transformer', self.numeric_transformer),
            ('categorical_transformer', self.categorical_transformer)
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X) 
        return self
    
    def transform(self, X):
        return self.pipeline.transform(X)
    
