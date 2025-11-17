"""
Hourly Weather Data Preprocessing Pipeline
==========================================
Production-ready preprocessing pipeline for Hanoi hourly weather data.

This module implements:
- Time series data splitting
- Variance-based feature selection
- Constant and duplicate removal
- High-cardinality categorical feature handling
- Domain-based outlier handling
- Missing value imputation strategies
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class TimeSeriesSplitter:
    """
    Clean train/val/test splitter for time-series forecasting.
    
    Maintains temporal order and prevents data leakage.
    """
    
    def __init__(self, datetime_col: str = "datetime"):
        """
        Initialize splitter.
        
        Args:
            datetime_col: Name of datetime column
        """
        self.datetime_col = datetime_col
    
    def split(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into train/val/test sets.
        
        Args:
            df: Input dataframe
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            
        Returns:
            Tuple of (train, val, test) dataframes
        """
        df = df.sort_values(self.datetime_col).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test


class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Remove features with variance below threshold.
    
    Features with zero or near-zero variance provide no predictive information.
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
        self.output_cols_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit selector on training data."""
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.numeric_cols = numeric_cols
        
        self.selector.fit(X[numeric_cols])
        
        self.retained_cols = list(numeric_cols[self.selector.get_support()])
        self.dropped_cols = [col for col in numeric_cols if col not in self.retained_cols]
        
        if self.dropped_cols:
            logger.info(f"Dropped {len(self.dropped_cols)} low-variance columns: {self.dropped_cols}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by removing low-variance features."""
        cols_to_keep = self.retained_cols + list(X.columns.difference(self.numeric_cols))
        self.output_cols_ = cols_to_keep
        return X[cols_to_keep]
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get feature names after transformation."""
        return np.array(self.output_cols_)


class ConstantAndDuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Remove constant features and duplicate rows.
    
    Constant features have zero variance and should be removed.
    Duplicate rows don't add information and can bias models.
    """
    
    def __init__(self):
        self.constant_cols = None
        self.output_cols_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify constant columns."""
        self.constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        
        if self.constant_cols:
            logger.info(f"Found {len(self.constant_cols)} constant columns: {self.constant_cols}")
        
        self.output_cols_ = [col for col in X.columns if col not in self.constant_cols]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant columns and duplicate rows."""
        X_cleaned = X.drop(columns=self.constant_cols, errors="ignore")
        
        n_before = len(X_cleaned)
        X_cleaned = X_cleaned.drop_duplicates()
        n_after = len(X_cleaned)
        
        if n_before != n_after:
            logger.info(f"Removed {n_before - n_after} duplicate rows")
        
        return X_cleaned
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get feature names after transformation."""
        return np.array(self.output_cols_)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers using domain knowledge and IQR method.
    
    Uses clipping instead of removal to:
    1. Preserve temporal continuity in time series
    2. Maintain valid extreme weather events
    3. Keep dataset size for modeling
    
    Physical constraints are applied where appropriate:
    - solarradiation: 0-1400 W/m²
    - precip: 0-50 mm/hour
    - winddir: 0-360°
    - sealevelpressure: 950-1050 hPa
    - humidity: 0-100%
    """
    
    def __init__(self, method: str = 'clip'):
        """
        Initialize handler.
        
        Args:
            method: Outlier handling method ('clip' or 'winsorize')
        """
        self.method = method
        self.bounds = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn outlier boundaries from training data."""
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Apply physical constraints per variable
            if col == 'solarradiation':
                lower, upper = 0, 1400
            elif col == 'precip':
                lower, upper = 0, 50
            elif col == 'winddir':
                lower, upper = 0, 360
            elif col == 'sealevelpressure':
                lower, upper = 950, 1050
            elif col == 'humidity':
                lower, upper = 0, 100
            elif col in ['windspeed', 'windgust']:
                lower = 0
                upper = X[col].quantile(0.99)  # 99th percentile
            else:
                # IQR method for other variables
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
            
            self.bounds[col] = (lower, upper)
        
        logger.info(f"Computed outlier bounds for {len(self.bounds)} numerical features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier handling."""
        X = X.copy()
        
        outliers_clipped = 0
        for col, (lower, upper) in self.bounds.items():
            if col in X.columns:
                n_outliers = ((X[col] < lower) | (X[col] > upper)).sum()
                outliers_clipped += n_outliers
                
                if self.method == 'clip':
                    X[col] = X[col].clip(lower=lower, upper=upper)
                elif self.method == 'winsorize':
                    X[col] = np.where(X[col] < lower, lower, X[col])
                    X[col] = np.where(X[col] > upper, upper, X[col])
        
        if outliers_clipped > 0:
            logger.info(f"Clipped {outliers_clipped} outlier values")
        
        return X


class MissingValueHandlerHourly(BaseEstimator, TransformerMixin):
    """
    Handle missing values using domain-specific strategies.
    
    Strategies:
    1. Drop: severerisk (81.7% missing)
    2. Forward/Backward fill: visibility, windspeed, winddir, sealevelpressure
       - Meteorological variables have temporal continuity
    3. Conditional imputation: precip
       - If precipprob = 0, then precip = 0
       - Otherwise: median
    4. Time-based + Interpolation: solarradiation, solarenergy, uvindex
       - Nighttime or heavy cloud → 0
       - Otherwise: linear interpolation
    5. KNN imputation: windgust
       - Uses correlation with other wind variables
    """
    
    def __init__(self, n_neighbors: int = 5):
        """
        Initialize handler.
        
        Args:
            n_neighbors: Number of neighbors for KNN imputation
        """
        self.n_neighbors = n_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        self.medians = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers on training data."""
        # Store median values
        for col in X.select_dtypes(include=[np.number]).columns:
            self.medians[col] = X[col].median()
        
        # Fit KNN imputer for wind variables
        wind_cols = ['windspeed', 'winddir', 'windgust']
        if all(col in X.columns for col in wind_cols):
            self.knn_imputer.fit(X[wind_cols])
            logger.info("Fitted KNN imputer for wind variables")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation strategies."""
        X = X.copy()
        
        # Log initial missing values
        initial_missing = X.isnull().sum().sum()
        
        # 1. Drop severerisk (excessive missing rate)
        if 'severerisk' in X.columns:
            X.drop(columns=['severerisk'], inplace=True)
            logger.info("Dropped 'severerisk' column (81.7% missing)")
        
        # 2. Forward/Backward fill for time series features
        time_cols = ['visibility', 'windspeed', 'winddir', 'sealevelpressure']
        for col in time_cols:
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(method='ffill').fillna(method='bfill')
        
        # 3. Conditional precipitation imputation
        if 'precip' in X.columns:
            if 'precipprob' in X.columns:
                # If no rain probability, then no precipitation
                X.loc[X['precip'].isna() & (X['precipprob'] == 0), 'precip'] = 0
            # Fill remaining with median
            X['precip'].fillna(self.medians.get('precip', 0), inplace=True)
        
        # 4. Solar features: time-based + interpolation
        solar_cols = ['solarradiation', 'solarenergy', 'uvindex']
        for col in solar_cols:
            if col in X.columns and X[col].isnull().any():
                # Nighttime values = 0
                if 'is_night' in X.columns:
                    X.loc[(X[col].isna()) & (X['is_night'] == 1), col] = 0
                # Interpolate remaining
                X[col] = X[col].interpolate(method='linear').fillna(0)
        
        # 5. KNN imputation for wind variables
        wind_cols = ['windspeed', 'winddir', 'windgust']
        if all(col in X.columns for col in wind_cols):
            if X[wind_cols].isnull().any().any():
                X[wind_cols] = self.knn_imputer.transform(X[wind_cols])
        
        # Log final missing values
        final_missing = X.isnull().sum().sum()
        logger.info(f"Missing values reduced: {initial_missing} → {final_missing}")
        
        return X


class HourlyDataPreprocessor:
    """
    Complete preprocessing pipeline for hourly weather data.
    
    Orchestrates all preprocessing steps in production-ready manner.
    """
    
    def __init__(
        self,
        target_col: str = 'temp',
        datetime_col: str = 'datetime',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        high_cardinality_threshold: int = 50,
        corr_threshold: float = 0.95
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_col: Name of target variable
            datetime_col: Name of datetime column
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            high_cardinality_threshold: Max unique values for categorical features
            corr_threshold: Correlation threshold for feature removal
        """
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.high_cardinality_threshold = high_cardinality_threshold
        self.corr_threshold = corr_threshold
        
        # Initialize pipeline components
        self.splitter = TimeSeriesSplitter(datetime_col=datetime_col)
        self.cleaning_pipeline = None
        self.outlier_handler = None
        self.mv_handler = None
        
        # Store feature lists
        self.dropped_features = []
        self.final_features = []
    
    def _remove_high_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with highly correlated features removed
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.corr_threshold)
        ]
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
            self.dropped_features.extend(to_drop)
            X = X.drop(columns=to_drop, errors='ignore')
        
        return X
    
    def _remove_high_cardinality(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove high-cardinality categorical features.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with high-cardinality features removed
        """
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        high_card_cols = [
            col for col in categorical_cols 
            if X[col].nunique() > self.high_cardinality_threshold
        ]
        
        # Also remove specific problematic columns
        cols_to_remove = list(set(high_card_cols) | {'stations', 'conditions', 'datetime'})
        cols_to_remove = [col for col in cols_to_remove if col in X.columns]
        
        if cols_to_remove:
            logger.info(f"Dropping high-cardinality/problematic columns: {cols_to_remove}")
            self.dropped_features.extend(cols_to_remove)
            X = X.drop(columns=cols_to_remove, errors='ignore')
        
        return X
    
    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Raw input dataframe
            
        Returns:
            Tuple of (train, val, test) processed dataframes
        """
        logger.info("="*60)
        logger.info("Starting Hourly Data Preprocessing Pipeline")
        logger.info("="*60)
        
        # 1. Split data
        logger.info("\n[1/6] Splitting data...")
        train, val, test = self.splitter.split(df, self.train_ratio, self.val_ratio)
        
        # 2. Separate features and target
        logger.info("\n[2/6] Separating features and target...")
        y_train = train[self.target_col]
        X_train = train.drop(columns=self.target_col)
        
        y_val = val[self.target_col]
        X_val = val.drop(columns=self.target_col)
        
        y_test = test[self.target_col]
        X_test = test.drop(columns=self.target_col)
        
        # Store datetime before processing
        train_datetime = X_train[self.datetime_col].reset_index(drop=True)
        val_datetime = X_val[self.datetime_col].reset_index(drop=True)
        test_datetime = X_test[self.datetime_col].reset_index(drop=True)
        
        # 3. Basic cleaning (variance, constants, duplicates)
        logger.info("\n[3/6] Applying basic cleaning...")
        self.cleaning_pipeline = Pipeline(steps=[
            ("var_thresh", VarianceThresholdSelector(threshold=0)),
            ("const_dup", ConstantAndDuplicateRemover())
        ])
        
        X_train = self.cleaning_pipeline.fit_transform(X_train)
        X_val = self.cleaning_pipeline.transform(X_val)
        X_test = self.cleaning_pipeline.transform(X_test)
        
        # 4. Remove highly correlated and high-cardinality features
        logger.info("\n[4/6] Removing correlated and high-cardinality features...")
        X_train = self._remove_high_correlation(X_train)
        X_train = self._remove_high_cardinality(X_train)
        
        # Apply same drops to val and test
        X_val = X_val.drop(columns=self.dropped_features, errors='ignore')
        X_test = X_test.drop(columns=self.dropped_features, errors='ignore')
        
        logger.info(f"Shape after feature removal - Train: {X_train.shape}")
        
        # 5. Handle outliers
        logger.info("\n[5/6] Handling outliers...")
        self.outlier_handler = OutlierHandler(method='clip')
        X_train = self.outlier_handler.fit_transform(X_train)
        X_val = self.outlier_handler.transform(X_val)
        X_test = self.outlier_handler.transform(X_test)
        
        # 6. Handle missing values
        logger.info("\n[6/6] Handling missing values...")
        self.mv_handler = MissingValueHandlerHourly(n_neighbors=5)
        X_train = self.mv_handler.fit_transform(X_train)
        X_val = self.mv_handler.transform(X_val)
        X_test = self.mv_handler.transform(X_test)
        
        # Store final features
        self.final_features = X_train.columns.tolist()
        
        # Combine datetime, features, and target
        train_processed = pd.concat([
            train_datetime, 
            X_train.reset_index(drop=True), 
            y_train.reset_index(drop=True)
        ], axis=1)
        
        val_processed = pd.concat([
            val_datetime,
            X_val.reset_index(drop=True),
            y_val.reset_index(drop=True)
        ], axis=1)
        
        test_processed = pd.concat([
            test_datetime,
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True)
        ], axis=1)
        
        logger.info("\n" + "="*60)
        logger.info("Preprocessing Complete!")
        logger.info(f"Final shapes - Train: {train_processed.shape}, Val: {val_processed.shape}, Test: {test_processed.shape}")
        logger.info(f"Total features: {len(self.final_features)}")
        logger.info(f"Dropped features: {len(self.dropped_features)}")
        logger.info("="*60 + "\n")
        
        return train_processed, val_processed, test_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input dataframe
            
        Returns:
            Processed dataframe
        """
        if self.cleaning_pipeline is None:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        # Separate features and target
        has_target = self.target_col in df.columns
        if has_target:
            y = df[self.target_col]
            X = df.drop(columns=self.target_col)
        else:
            X = df.copy()
        
        # Store datetime
        datetime_col = X[self.datetime_col].reset_index(drop=True) if self.datetime_col in X.columns else None
        
        # Apply transformations
        X = self.cleaning_pipeline.transform(X)
        X = X.drop(columns=self.dropped_features, errors='ignore')
        X = self.outlier_handler.transform(X)
        X = self.mv_handler.transform(X)
        
        # Combine back
        if datetime_col is not None:
            X = pd.concat([datetime_col, X.reset_index(drop=True)], axis=1)
        
        if has_target:
            X = pd.concat([X, y.reset_index(drop=True)], axis=1)
        
        return X
    
    def save_processed_data(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str = "../../dataset/processed"
    ) -> None:
        """
        Save processed data to CSV files.
        
        Args:
            train: Training dataframe
            val: Validation dataframe
            test: Test dataframe
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train_hourly_processed.csv"
        val_path = output_path / "val_hourly_processed.csv"
        test_path = output_path / "test_hourly_processed.csv"
        
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        
        logger.info(f"\n✅ Saved processed files:")
        logger.info(f"   - {train_path}")
        logger.info(f"   - {val_path}")
        logger.info(f"   - {test_path}")


def main():
    """Main execution function."""
    # Load raw data
    logger.info("Loading raw hourly data...")
    data_path = Path("../../dataset/raw/Hanoi Hourly.csv")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Initialize and run preprocessor
    preprocessor = HourlyDataPreprocessor(
        target_col='temp',
        datetime_col='datetime',
        train_ratio=0.8,
        val_ratio=0.1,
        high_cardinality_threshold=50,
        corr_threshold=0.95
    )
    
    # Process data
    train, val, test = preprocessor.fit_transform(df)
    
    # Save processed data
    preprocessor.save_processed_data(train, val, test)
    
    logger.info("\n✅ Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
