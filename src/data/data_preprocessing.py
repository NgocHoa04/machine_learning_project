import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .data_helper import *          

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
        Clean feature name by:
        - Replace commas and spaces with '_'
        - Remove/replace other special characters with '_'
        """
        name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        return name

    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)

        raw_feature_names = self.preprocessor.get_feature_names_out()
        # Save mapping for later reference if needed
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

        # Convert numerical columns to float if needed
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='ignore')
        
        return data_df

class PrecipImputer(BaseEstimator, TransformerMixin):
    def __init__(self, precip_col="precip", precipprob_col="precipprob", short_limit=6):
        self.precip_col = precip_col
        self.precipprob_col = precipprob_col
        self.short_limit = short_limit
        self.imputer = None

    def fit(self, X, y=None):
        df = X.copy()
        if self.precipprob_col in df.columns:
            mask_zero = df[self.precip_col].isna() & (df[self.precipprob_col] == 0)
            df.loc[mask_zero, self.precip_col] = 0.0
        df[self.precip_col] = df[self.precip_col].interpolate(method="time", limit=self.short_limit)
        cols = [c for c in [self.precip_col, "cloudcover", "humidity", self.precipprob_col, "windspeed"] if c in df.columns]
        if len(cols) > 1:
            self.imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0), max_iter=10)
            self.imputer.fit(df[cols])
        return self

    def transform(self, X):
        df = X.copy()
        if self.precipprob_col in df.columns:
            mask_zero = df[self.precip_col].isna() & (df[self.precipprob_col] == 0)
            df.loc[mask_zero, self.precip_col] = 0.0
        df[self.precip_col] = df[self.precip_col].interpolate(method="time", limit=self.short_limit)
        if self.imputer:
            cols = self.imputer.feature_names_in_
            df[cols] = self.imputer.transform(df[cols])
        return df
    

class WinddirImputer(BaseEstimator, TransformerMixin):
    def __init__(self, winddir_col="winddir", windspeed_col="windspeed", short_limit=6, knn_neighbors=5):
        self.winddir_col = winddir_col
        self.windspeed_col = windspeed_col
        self.short_limit = short_limit
        self.knn_neighbors = knn_neighbors
        self.imputer = None

    def fit(self, X, y=None):
        df = X.copy()
        wd_rad = np.deg2rad(df[self.winddir_col])
        df["wd_sin"] = np.sin(wd_rad)
        df["wd_cos"] = np.cos(wd_rad)
        df["wd_sin"] = df["wd_sin"].interpolate(method="time", limit=self.short_limit)
        df["wd_cos"] = df["wd_cos"].interpolate(method="time", limit=self.short_limit)
        cols = ["wd_sin", "wd_cos"]
        if self.windspeed_col in df.columns:
            cols.append(self.windspeed_col)
        self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        self.imputer.fit(df[cols])
        return self

    def transform(self, X):
        df = X.copy()
        wd_rad = np.deg2rad(df[self.winddir_col])
        df["wd_sin"] = np.sin(wd_rad)
        df["wd_cos"] = np.cos(wd_rad)
        df["wd_sin"] = df["wd_sin"].interpolate(method="time", limit=self.short_limit)
        df["wd_cos"] = df["wd_cos"].interpolate(method="time", limit=self.short_limit)
        cols = ["wd_sin", "wd_cos"]
        if self.windspeed_col in df.columns:
            cols.append(self.windspeed_col)
        df[cols] = self.imputer.transform(df[cols])
        df[self.winddir_col] = (np.rad2deg(np.arctan2(df["wd_sin"], df["wd_cos"])) + 360) % 360
        return df
    
class SolarradiationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, sol_col="solarradiation", lat=21.0278, short_limit=6):
        self.sol_col = sol_col
        self.lat = lat
        self.short_limit = short_limit
        self.imputer = None

    def solar_elevation(self, index):
        doy = index.dayofyear.values
        hour = index.hour + index.minute / 60.0
        decl = 23.44 * np.sin(2 * np.pi * (284 + doy) / 365)
        decl_rad = np.deg2rad(decl)
        lat_rad = np.deg2rad(self.lat)
        hour_angle = np.deg2rad((hour - 12) * 15)
        elev = np.arcsin(np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle))
        return np.rad2deg(elev)

    def fit(self, X, y=None):
        df = X.copy()
        df["solar_elev"] = self.solar_elevation(df.index)
        df.loc[df["solar_elev"] <= 0, self.sol_col] = 0.0
        df.loc[df["solar_elev"] > 0, self.sol_col] = df.loc[df["solar_elev"] > 0, self.sol_col].interpolate(method="time", limit=self.short_limit)
        cols = [c for c in [self.sol_col, "cloudcover", "uvindex", "solarenergy", "humidity"] if c in df.columns]
        if len(cols) > 1:
            self.imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0), max_iter=10)
            self.imputer.fit(df[cols])
        return self

    def transform(self, X):
        df = X.copy()
        df["solar_elev"] = self.solar_elevation(df.index)
        df.loc[df["solar_elev"] <= 0, self.sol_col] = 0.0
        df.loc[df["solar_elev"] > 0, self.sol_col] = df.loc[df["solar_elev"] > 0, self.sol_col].interpolate(method="time", limit=self.short_limit)
        if self.imputer:
            cols = self.imputer.feature_names_in_
            df[cols] = self.imputer.transform(df[cols])
        return df
    
impute_pipeline = Pipeline([
    ("impute_precip", PrecipImputer()),
    ("impute_winddir", WinddirImputer()),
    ("impute_solar", SolarradiationImputer())
])