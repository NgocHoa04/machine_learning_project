import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .data_helper import *          
import re

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
class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    'onehot',
                    OneHotEncoder(
                        handle_unknown='ignore',
                        drop='first',
                        sparse_output=False
                    ),
                    self.categorical_features
                )
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

    @staticmethod
    def _clean_name(name: str) -> str:
        """
        Làm sạch tên cột:
        - Thay dấu phẩy và khoảng trắng bằng '_'
        - Bỏ / thay các ký tự đặc biệt khác thành '_'
        """
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        # Nếu muốn chặt chẽ hơn thì:
        name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        return name

    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)

        raw_feature_names = self.preprocessor.get_feature_names_out()
        # Lưu cả mapping nếu sau này cần tra lại
        self.feature_name_map_ = {
            raw: self._clean_name(raw) for raw in raw_feature_names
        }
        self.feature_names_out_ = np.array(
            [self.feature_name_map_[raw] for raw in raw_feature_names]
        )
        return self
    
    def transform(self, X):
        data_numpy = self.preprocessor.transform(X)

        data_df = pd.DataFrame(
            data_numpy, 
            columns=self.feature_names_out_,
            index=X.index
        )

        # Convert numerical columns to float nếu cần
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='ignore')
        
        return data_df