"""
Daily Weather Data Preprocessing Pipeline
=========================================
Production-ready preprocessing pipeline for Hanoi daily weather data.

This module implements:
- Variance-based feature selection
- Constant and duplicate removal
- Categorical feature encoding
- Advanced imputation strategies for meteorological variables
- Data transformation and scaling
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

from .data_helper import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Remove features with variance below threshold.
    
    Features with zero or near-zero variance provide no predictive information
    and can cause issues in model training.
    """
    
    def __init__(self, threshold: float = 0):
        """
        Initialize selector.
        
        Args:
            threshold: Minimum variance threshold
        """
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.numeric_cols = None
        self.retained_cols = None
        self.dropped_cols = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit selector on training data.
        
        Args:
            X: Input dataframe
            y: Target variable (not used)
            
        Returns:
            self
        """
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.numeric_cols = numeric_cols
        self.selector.fit(X[numeric_cols])
        
        # Save retained and dropped columns
        self.retained_cols = numeric_cols[self.selector.get_support()]
        self.dropped_cols = [col for col in numeric_cols if col not in self.retained_cols]
        
        if self.dropped_cols:
            logger.info(f"Dropped {len(self.dropped_cols)} low-variance columns: {self.dropped_cols}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by removing low-variance features.
        
        Args:
            X: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        cols_to_keep = list(self.retained_cols) + list(X.columns.difference(self.numeric_cols))
        return X[cols_to_keep]


class ConstantAndDuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Remove constant features and duplicate rows.
    
    Constant features (single unique value) have zero variance and provide
    no information. Duplicate rows don't add information and can bias models.
    """
    
    def __init__(self):
        self.constant_cols = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Identify constant columns.
        
        Args:
            X: Input dataframe
            y: Target variable (not used)
            
        Returns:
            self
        """
        self.constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        
        if self.constant_cols:
            logger.info(f"Found {len(self.constant_cols)} constant columns: {self.constant_cols}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant columns and duplicate rows.
        
        Args:
            X: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Remove constant columns
        X_cleaned = X.drop(columns=self.constant_cols, errors="ignore")
        
        # Remove duplicate rows
        n_before = len(X_cleaned)
        X_cleaned = X_cleaned.drop_duplicates()
        n_after = len(X_cleaned)
        
        if n_before != n_after:
            logger.info(f"Removed {n_before - n_after} duplicate rows")
        
        return X_cleaned


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    Transform categorical features using one-hot encoding.
    
    Handles unknown categories and produces clean feature names
    compatible with model training frameworks.
    """
    
    def __init__(self, categorical_features: List[str]):
        """
        Initialize transformer.
        
        Args:
            categorical_features: List of categorical feature names to encode
        """
        self.categorical_features = categorical_features
        self.preprocessor = None
        self.feature_name_map_ = None
        self.feature_names_out_ = None
        
        # Initialize column transformer
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
        Clean feature name by replacing special characters.
        
        Args:
            name: Original feature name
            
        Returns:
            Cleaned feature name
        """
        name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        return name

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit transformer on training data.
        
        Args:
            X: Input dataframe
            y: Target variable (not used)
            
        Returns:
            self
        """
        self.preprocessor.fit(X, y)

        # Get and clean feature names
        raw_feature_names = self.preprocessor.get_feature_names_out()
        self.feature_name_map_ = {
            raw: self._clean_name(raw) for raw in raw_feature_names
        }
        self.feature_names_out_ = np.array(
            [self.feature_name_map_[raw] for raw in raw_feature_names]
        )
        
        logger.info(f"Transformed {len(self.categorical_features)} categorical features into {len(self.feature_names_out_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features to one-hot encoded format.
        
        Args:
            X: Input dataframe
            
        Returns:
            Transformed dataframe with clean feature names
        """
        data_numpy = self.preprocessor.transform(X)

        data_df = pd.DataFrame(
            data_numpy, 
            columns=self.feature_names_out_,
            index=X.index
        )

        # Convert numerical columns to proper type
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='ignore')
        
        return data_df


class PrecipImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing precipitation values using domain knowledge.
    
    Strategy:
    1. If precipprob = 0, set precip = 0
    2. Interpolate short gaps (≤ short_limit days)
    3. Use IterativeImputer with RandomForest for remaining gaps
       based on related features (cloudcover, humidity, windspeed)
    """
    
    def __init__(
        self, 
        precip_col: str = "precip", 
        precipprob_col: str = "precipprob", 
        short_limit: int = 6,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 0
    ):
        """
        Initialize imputer.
        
        Args:
            precip_col: Name of precipitation column
            precipprob_col: Name of precipitation probability column
            short_limit: Maximum gap length for interpolation
            n_estimators: Number of trees in RandomForest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.precip_col = precip_col
        self.precipprob_col = precipprob_col
        self.short_limit = short_limit
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.imputer = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit imputer on training data.
        
        Args:
            X: Input dataframe with datetime index
            y: Target variable (not used)
            
        Returns:
            self
        """
        df = X.copy()
        
        # Step 1: Zero precipitation when probability is zero
        if self.precipprob_col in df.columns:
            mask_zero = df[self.precip_col].isna() & (df[self.precipprob_col] == 0)
            df.loc[mask_zero, self.precip_col] = 0.0
            n_filled = mask_zero.sum()
            if n_filled > 0:
                logger.info(f"Set {n_filled} precip values to 0 where precipprob = 0")
        
        # Step 2: Interpolate short gaps
        df[self.precip_col] = df[self.precip_col].interpolate(
            method="time", 
            limit=self.short_limit
        )
        
        # Step 3: Fit IterativeImputer on related features
        cols = [
            c for c in [self.precip_col, "cloudcover", "humidity", 
                       self.precipprob_col, "windspeed"] 
            if c in df.columns
        ]
        
        if len(cols) > 1:
            self.imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    n_jobs=-1,
                    random_state=self.random_state
                ),
                max_iter=10,
                random_state=self.random_state
            )
            self.imputer.fit(df[cols])
            logger.info(f"Fitted IterativeImputer for {self.precip_col} using {len(cols)} features")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply precipitation imputation.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with imputed precipitation values
        """
        df = X.copy()
        
        # Apply same steps as fit
        if self.precipprob_col in df.columns:
            mask_zero = df[self.precip_col].isna() & (df[self.precipprob_col] == 0)
            df.loc[mask_zero, self.precip_col] = 0.0
        
        df[self.precip_col] = df[self.precip_col].interpolate(
            method="time", 
            limit=self.short_limit
        )
        
        if self.imputer:
            cols = self.imputer.feature_names_in_
            df[cols] = self.imputer.transform(df[cols])
        
        return df


class WinddirImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing wind direction values using circular statistics.
    
    Wind direction is circular (0° = 360°), so we convert to sin/cos
    components for proper interpolation and imputation.
    
    Strategy:
    1. Convert wind direction to sin/cos components
    2. Interpolate short gaps in sin/cos space
    3. Use KNN imputation for remaining gaps (considering windspeed)
    4. Convert back to degrees
    """
    
    def __init__(
        self, 
        winddir_col: str = "winddir", 
        windspeed_col: str = "windspeed", 
        short_limit: int = 6, 
        knn_neighbors: int = 5
    ):
        """
        Initialize imputer.
        
        Args:
            winddir_col: Name of wind direction column
            windspeed_col: Name of wind speed column
            short_limit: Maximum gap length for interpolation
            knn_neighbors: Number of neighbors for KNN imputation
        """
        self.winddir_col = winddir_col
        self.windspeed_col = windspeed_col
        self.short_limit = short_limit
        self.knn_neighbors = knn_neighbors
        self.imputer = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit imputer on training data.
        
        Args:
            X: Input dataframe with datetime index
            y: Target variable (not used)
            
        Returns:
            self
        """
        df = X.copy()
        
        # Convert to circular components
        wd_rad = np.deg2rad(df[self.winddir_col])
        df["wd_sin"] = np.sin(wd_rad)
        df["wd_cos"] = np.cos(wd_rad)
        
        # Interpolate short gaps
        df["wd_sin"] = df["wd_sin"].interpolate(method="time", limit=self.short_limit)
        df["wd_cos"] = df["wd_cos"].interpolate(method="time", limit=self.short_limit)
        
        # Prepare columns for KNN
        cols = ["wd_sin", "wd_cos"]
        if self.windspeed_col in df.columns:
            cols.append(self.windspeed_col)
        
        # Fit KNN imputer
        self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        self.imputer.fit(df[cols])
        
        logger.info(f"Fitted KNN imputer for {self.winddir_col} using {len(cols)} features")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply wind direction imputation.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with imputed wind direction values
        """
        df = X.copy()
        
        # Convert to circular components
        wd_rad = np.deg2rad(df[self.winddir_col])
        df["wd_sin"] = np.sin(wd_rad)
        df["wd_cos"] = np.cos(wd_rad)
        
        # Interpolate
        df["wd_sin"] = df["wd_sin"].interpolate(method="time", limit=self.short_limit)
        df["wd_cos"] = df["wd_cos"].interpolate(method="time", limit=self.short_limit)
        
        # KNN imputation
        cols = ["wd_sin", "wd_cos"]
        if self.windspeed_col in df.columns:
            cols.append(self.windspeed_col)
        df[cols] = self.imputer.transform(df[cols])
        
        # Convert back to degrees
        df[self.winddir_col] = (np.rad2deg(np.arctan2(df["wd_sin"], df["wd_cos"])) + 360) % 360
        
        return df


class SolarradiationImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing solar radiation values using astronomical calculations.
    
    Solar radiation depends on sun elevation angle, which can be calculated
    from date/time and latitude. When sun is below horizon, radiation = 0.
    
    Strategy:
    1. Calculate solar elevation angle from datetime
    2. Set radiation = 0 when sun is below horizon
    3. Interpolate short daytime gaps
    4. Use IterativeImputer with related features (cloudcover, uvindex, etc.)
    """
    
    def __init__(
        self, 
        sol_col: str = "solarradiation", 
        lat: float = 21.0278,  # Hanoi latitude
        short_limit: int = 6,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 0
    ):
        """
        Initialize imputer.
        
        Args:
            sol_col: Name of solar radiation column
            lat: Latitude in degrees (default: Hanoi)
            short_limit: Maximum gap length for interpolation
            n_estimators: Number of trees in RandomForest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.sol_col = sol_col
        self.lat = lat
        self.short_limit = short_limit
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.imputer = None

    def solar_elevation(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Calculate solar elevation angle.
        
        Uses simplified formula based on day of year and hour.
        
        Args:
            index: Pandas DatetimeIndex
            
        Returns:
            Array of solar elevation angles in degrees
        """
        doy = index.dayofyear.values
        hour = index.hour + index.minute / 60.0
        
        # Solar declination angle
        decl = 23.44 * np.sin(2 * np.pi * (284 + doy) / 365)
        decl_rad = np.deg2rad(decl)
        lat_rad = np.deg2rad(self.lat)
        
        # Hour angle
        hour_angle = np.deg2rad((hour - 12) * 15)
        
        # Solar elevation
        elev = np.arcsin(
            np.sin(lat_rad) * np.sin(decl_rad) + 
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle)
        )
        
        return np.rad2deg(elev)

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit imputer on training data.
        
        Args:
            X: Input dataframe with datetime index
            y: Target variable (not used)
            
        Returns:
            self
        """
        df = X.copy()
        
        # Calculate solar elevation
        df["solar_elev"] = self.solar_elevation(df.index)
        
        # Set nighttime values to 0
        n_night = (df["solar_elev"] <= 0).sum()
        df.loc[df["solar_elev"] <= 0, self.sol_col] = 0.0
        logger.info(f"Set {n_night} nighttime {self.sol_col} values to 0")
        
        # Interpolate daytime gaps
        df.loc[df["solar_elev"] > 0, self.sol_col] = df.loc[
            df["solar_elev"] > 0, self.sol_col
        ].interpolate(method="time", limit=self.short_limit)
        
        # Fit IterativeImputer on related features
        cols = [
            c for c in [self.sol_col, "cloudcover", "uvindex", 
                       "solarenergy", "humidity"] 
            if c in df.columns
        ]
        
        if len(cols) > 1:
            self.imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    n_jobs=-1,
                    random_state=self.random_state
                ),
                max_iter=10,
                random_state=self.random_state
            )
            self.imputer.fit(df[cols])
            logger.info(f"Fitted IterativeImputer for {self.sol_col} using {len(cols)} features")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply solar radiation imputation.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with imputed solar radiation values
        """
        df = X.copy()
        
        # Calculate solar elevation
        df["solar_elev"] = self.solar_elevation(df.index)
        
        # Set nighttime values to 0
        df.loc[df["solar_elev"] <= 0, self.sol_col] = 0.0
        
        # Interpolate daytime gaps
        df.loc[df["solar_elev"] > 0, self.sol_col] = df.loc[
            df["solar_elev"] > 0, self.sol_col
        ].interpolate(method="time", limit=self.short_limit)
        
        # Apply IterativeImputer
        if self.imputer:
            cols = self.imputer.feature_names_in_
            df[cols] = self.imputer.transform(df[cols])
        
        return df


class DailyDataPreprocessor:
    """
    Complete preprocessing pipeline for daily weather data.
    
    Orchestrates all preprocessing steps in production-ready manner.
    Includes variance selection, encoding, and specialized imputation.
    """
    
    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        variance_threshold: float = 0.0,
        short_limit: int = 6,
        knn_neighbors: int = 5,
        latitude: float = 21.0278
    ):
        """
        Initialize preprocessor.
        
        Args:
            categorical_features: List of categorical feature names
            variance_threshold: Minimum variance threshold
            short_limit: Maximum gap for time interpolation
            knn_neighbors: Number of neighbors for KNN imputation
            latitude: Latitude for solar calculations (default: Hanoi)
        """
        self.categorical_features = categorical_features or []
        self.variance_threshold = variance_threshold
        self.short_limit = short_limit
        self.knn_neighbors = knn_neighbors
        self.latitude = latitude
        
        # Initialize pipelines
        self.cleaning_pipeline = None
        self.impute_pipeline = None
        self.transform_pipeline = None
        
    def build_pipelines(self):
        """Build preprocessing pipelines."""
        # Cleaning pipeline
        self.cleaning_pipeline = Pipeline(steps=[
            ('constant_and_duplicate_remover', ConstantAndDuplicateRemover()),
            ('variance_threshold_selector', VarianceThresholdSelector(
                threshold=self.variance_threshold
            ))
        ])
        
        # Imputation pipeline
        self.impute_pipeline = Pipeline([
            ("impute_precip", PrecipImputer(short_limit=self.short_limit)),
            ("impute_winddir", WinddirImputer(
                short_limit=self.short_limit,
                knn_neighbors=self.knn_neighbors
            )),
            ("impute_solar", SolarradiationImputer(
                lat=self.latitude,
                short_limit=self.short_limit
            ))
        ])
        
        logger.info("Built preprocessing pipelines")
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y=None
    ) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Input dataframe
            y: Target variable (not used)
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("="*60)
        logger.info("Starting Daily Data Preprocessing Pipeline")
        logger.info("="*60)
        
        # Build pipelines if not already built
        if self.cleaning_pipeline is None:
            self.build_pipelines()
        
        # Step 1: Cleaning
        logger.info("\n[1/3] Applying cleaning pipeline...")
        X_cleaned = self.cleaning_pipeline.fit_transform(X)
        logger.info(f"Shape after cleaning: {X_cleaned.shape}")
        
        # Step 2: Imputation
        logger.info("\n[2/3] Applying imputation pipeline...")
        X_imputed = self.impute_pipeline.fit_transform(X_cleaned)
        logger.info(f"Shape after imputation: {X_imputed.shape}")
        
        # Step 3: Categorical encoding (if applicable)
        if self.categorical_features:
            logger.info("\n[3/3] Applying categorical encoding...")
            valid_cats = [c for c in self.categorical_features if c in X_imputed.columns]
            if valid_cats:
                self.transform_pipeline = DataTransformer(valid_cats)
                X_transformed = self.transform_pipeline.fit_transform(X_imputed)
            else:
                logger.warning("No valid categorical features found")
                X_transformed = X_imputed
        else:
            logger.info("\n[3/3] Skipping categorical encoding (no features specified)")
            X_transformed = X_imputed
        
        logger.info("\n" + "="*60)
        logger.info("Preprocessing Complete!")
        logger.info(f"Final shape: {X_transformed.shape}")
        logger.info("="*60 + "\n")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        if self.cleaning_pipeline is None:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        # Apply pipelines
        X_cleaned = self.cleaning_pipeline.transform(X)
        X_imputed = self.impute_pipeline.transform(X_cleaned)
        
        if self.transform_pipeline is not None:
            X_transformed = self.transform_pipeline.transform(X_imputed)
        else:
            X_transformed = X_imputed
        
        return X_transformed


# Legacy pipeline objects for backward compatibility
remove_low_variance_pipeline = Pipeline(steps=[
    ('constant_and_duplicate_remover', ConstantAndDuplicateRemover()),
    ('variance_threshold_selector', VarianceThresholdSelector(threshold=0.0))
])

impute_pipeline = Pipeline([
    ("impute_precip", PrecipImputer()),
    ("impute_winddir", WinddirImputer()),
    ("impute_solar", SolarradiationImputer())
])


def main():
    """Main execution function for testing."""
    logger.info("Daily preprocessing module loaded successfully")
    logger.info("Available classes:")
    logger.info("  - VarianceThresholdSelector")
    logger.info("  - ConstantAndDuplicateRemover")
    logger.info("  - DataTransformer")
    logger.info("  - PrecipImputer")
    logger.info("  - WinddirImputer")
    logger.info("  - SolarradiationImputer")
    logger.info("  - DailyDataPreprocessor")


if __name__ == "__main__":
    main()
