import numpy as np
import pandas as pd
import argparse
import os
from typing import Sequence, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

class HanoiDailyFE(BaseEstimator, TransformerMixin):
    """
    Feature engineering for Hanoi weather data.
    Designed to work inside an sklearn Pipeline (implements fit/transform).
    """

    def __init__(
        self,
        date_col: str = "datetime",
        sunrise_col: str = "sunrise",
        sunset_col: str = "sunset",
        winddir_col: str = "winddir",
        windsp_col: str = "windspeed",
        lag_days: Sequence[int] = (1, 2, 3, 7),
        roll_windows: Sequence[int] = (3, 7, 14, 60, 90),
        calm_thr: float = 0.5,
        lag_base_cols: Optional[Sequence[str]] = None,
        dropna: bool = False,  # if True, drop NaNs after generating lag/rolling features
    ) -> None:
        self.date_col = date_col
        self.sunrise_col = sunrise_col
        self.sunset_col = sunset_col
        self.winddir_col = winddir_col
        self.windsp_col = windsp_col
        self.lag_days = tuple(lag_days)
        self.roll_windows = tuple(roll_windows)
        self.calm_thr = calm_thr
        self.dropna = dropna

        # Columns used to generate lag/rolling features
        if lag_base_cols is None:
            self.lag_base_cols = [
                "humidity",
                "windspeed",
                "precip",
                "solarradiation",
                "cloudcover",
                "precipcover",
                "daylength_hours",
                "dew",
            ]
        else:
            self.lag_base_cols = list(lag_base_cols)
    
    @staticmethod
    def load_hanoi_daily_data(file_path: str) -> pd.DataFrame:
        """
        Load Hanoi daily weather data from CSV file.
        Assumes the CSV has a 'datetime' column in YYYY-MM-DD format.
        """
        df = pd.read_csv(file_path)
        return df

    # ===================== 1. MONSOON ZONES & WIND FEATURES =====================

    @staticmethod
    def _monsoon_zone(deg: Union[float, int, None]) -> str:
        """
        Classify wind direction into monsoon zones (Hanoi-specific):

        - NE  : 20–80°   (Northeast monsoon: winter, cool/humid)
        - SW  : 200–260° (Southwest monsoon: summer, hot/humid/stormy)
        - Other / Unknown: all other angles or missing values
        """
        if pd.isna(deg):
            return "Unknown"

        d = float(deg) % 360  # normalize into [0, 360)

        if 20.0 <= d <= 80.0:
            return "NE"
        if 200.0 <= d <= 260.0:
            return "SW"
        return "Other"

    def _fe_wind_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create wind-related features from wind direction (and wind speed if available):

        - Monsoon zone one-hot:
            monsoon_NE, monsoon_SW, monsoon_Other
        - winddir_sin, winddir_cos (cyclical encoding of direction)
        - u_wind, v_wind (wind vector components, direction as 'from')
        - is_calm (calm wind if speed <= calm_thr)

        If wind direction column is missing, create zero/dummy features to
        keep the pipeline robust.
        """
        out = df.copy()

        if self.winddir_col not in out.columns:
            # No wind direction -> create dummy features to preserve structure
            out["monsoon_NE"] = 0
            out["monsoon_SW"] = 0
            out["monsoon_Other"] = 0
            out["winddir_sin"] = 0.0
            out["winddir_cos"] = 0.0
            out["u_wind"] = 0.0
            out["v_wind"] = 0.0
            out["is_calm"] = 0
            return out

        # 1) Monsoon zone one-hot encoding
        monsoon = out[self.winddir_col].apply(self._monsoon_zone)
        out["monsoon_NE"] = (monsoon == "NE").astype(int)
        out["monsoon_SW"] = (monsoon == "SW").astype(int)
        out["monsoon_Other"] = (monsoon == "Other").astype(int)

        # 2) Cyclical encoding (sin/cos) of wind direction
        rad = np.deg2rad(out[self.winddir_col].astype(float))
        out["winddir_sin"] = np.sin(rad)
        out["winddir_cos"] = np.cos(rad)

        # 3) Wind vector components (using speed if available)
        if self.windsp_col in out.columns:
            spd = out[self.windsp_col].astype(float).fillna(0.0)
        else:
            spd = pd.Series(0.0, index=out.index)

        # winddir is direction FROM which the wind blows:
        out["u_wind"] = -spd * np.sin(rad)
        out["v_wind"] = -spd * np.cos(rad)

        # 4) Calm wind indicator
        out["is_calm"] = (spd <= self.calm_thr).astype(int)

        for col in ["winddir_sin", "winddir_cos", "u_wind", "v_wind"]:
            out[col] = out[col].fillna(0.0)

        return out

    # ===================== 2. LAG & ROLLING FEATURES (NO LEAKAGE) =====================

    def _add_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag and rolling statistics for selected weather variables.

        - Data is sorted by date_col.
        - Lag features: col_lag{L} for each L in lag_days.
        - Rolling features (mean, std) on shifted series (shift(1)) to prevent leakage:
          col_roll{w}d_mean, col_roll{w}d_std for each window in roll_windows.
        """
        out = df.copy()
        out = out.sort_values(self.date_col)

        target_cols = [c for c in self.lag_base_cols if c in out.columns]

        for col in target_cols:
            # ---------- Lag features ----------
            for L in self.lag_days:
                out[f"{col}_lag{L}"] = out[col].shift(L)

            # ---------- Rolling features ----------
            for w in self.roll_windows:
                # 
                base = out[col].rolling(window=w)
                out[f"{col}_roll{w}d_mean"] = base.mean()
                out[f"{col}_roll{w}d_std"] = base.std()

        return out

    # ===================== 3. CALENDAR, DAYLENGTH & SEASONAL FEATURES =====================

    def _add_calendar_and_daylength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar and daylength-related features for Hanoi:

        - From datetime: month, dayofyear, day of week (dow)
        - Yearly cycle: doy_sin, doy_cos
        - daylength_hours = (sunset - sunrise) / 3600
        - Seasonal cycle (based on sunrise date): season_sin, season_cos
        """
        out = df.copy()

        out[self.date_col] = pd.to_datetime(out[self.date_col])

        if self.sunrise_col in out.columns and self.sunset_col in out.columns:
            out["sunrise_dt"] = pd.to_datetime(out[self.sunrise_col])
            out["sunset_dt"] = pd.to_datetime(out[self.sunset_col])
        else:
            # If sunrise/sunset are missing, use date_col as dummy timestamps
            out["sunrise_dt"] = out[self.date_col]
            out["sunset_dt"] = out[self.date_col]

        # Basic calendar features
        out["month"] = out[self.date_col].dt.month
        out["dayofyear"] = out[self.date_col].dt.day_of_year
        out["dow"] = out[self.date_col].dt.dayofweek

        # Yearly cycle encoding
        doy = out["dayofyear"]
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Daylength in hours
        dl_seconds = (out["sunset_dt"] - out["sunrise_dt"]).dt.total_seconds()
        out["daylength_hours"] = dl_seconds / 3600.0

        # Seasonal cycle based on sunrise day-of-year
        out["day_of_year"] = out["sunrise_dt"].dt.dayofyear
        out["season_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
        out["season_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

        return out

    # ===================== 4. STANDARD SKLEARN API =====================

    def fit(self, X: pd.DataFrame, y=None):
        """
        No parameters are learned (pure feature engineering).
        Included for compatibility with sklearn Pipeline.
        """
        # You could add column presence checks here if desired.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full feature engineering pipeline:
        - Calendar & daylength features
        - Wind-related features
        - Lag & rolling statistics

        Returns a DataFrame with original (kept) columns plus engineered features.
        """
        out = X.copy()

        # 1) Calendar, daylength and seasonal features
        out = self._add_calendar_and_daylength_features(out)

        # 2) Wind features
        out = self._fe_wind_block(out)

        # 3) Lag & rolling features
        out = self._add_lag_rolling_features(out)

        # Drop intermediate / unnecessary columns
        to_drop = [
            "sunrise_dt",
            "sunset_dt",
            "day_of_year",
            self.winddir_col,
            self.sunrise_col,
            self.sunset_col,
        ]
        for col in set(to_drop):
            if col in out.columns:
                out = out.drop(columns=col)

        out = out.dropna().reset_index(drop=True)

        return out

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature Engineering for Hanoi Daily Weather Data')
    parser.add_argument('--input', type=str, required=False,
                        help='Input CSV file path (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, required=False,
                        help='Output CSV file path (optional, uses default if not provided)')
    
    args = parser.parse_args()
    
    # Set default paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_input = os.path.join(project_root, 'dataset', 'processed', 'Hanoi_Daily_Selected.csv')
    default_output = os.path.join(project_root, 'dataset', 'processed', 'Hanoi_daily_FE_full.csv')
    
    input_file = args.input if args.input else default_input
    output_file = args.output if args.output else default_output
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Apply Daily Feature Engineering
    print("\nApplying Daily Feature Engineering...")
    fe = HanoiDailyFE(
        date_col="datetime",
        sunrise_col="sunrise",
        sunset_col="sunset",
        winddir_col="winddir",
        windsp_col="windspeed",
        lag_days=(1, 2, 3, 7),
        roll_windows=(3, 7, 14, 60, 90),
        calm_thr=0.5,
        dropna=False
    )
    
    # Transform data
    df_transformed = fe.fit_transform(df)
    print(f"Transformed to {len(df_transformed)} rows with {len(df_transformed.columns)} columns")
    
    # Save output
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_transformed.to_csv(output_file, index=False)
    print("Done!")
    print(f"\nOutput saved: {output_file}")
    print(f"Shape: {df_transformed.shape}")
