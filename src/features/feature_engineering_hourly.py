import numpy as np
import pandas as pd
import argparse
import os
from typing import Sequence, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

class HanoiHourlyFE(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        date_col: str = "datetime",
        lat: float = 21.0278,
        winddir_col: str = "winddir",
        windsp_col: str = "windspeed",
        icon_col: str = "icon",
        calm_thr: float = 0.5,
        all_lags_hours: Sequence[int] = (1, 3, 12, 24),
        rolling_windows_hours: Sequence[int] = (3, 6, 12, 24, 72, 168),
        lag_base_cols: Optional[Sequence[str]] = None
    ) -> None:
        
        self.date_col = date_col
        self.lat = lat
        self.winddir_col = winddir_col
        self.windsp_col = windsp_col
        self.icon_col = icon_col 
        self.calm_thr = calm_thr
        self.all_lags = sorted(set(all_lags_hours))
        self.roll_windows = tuple(rolling_windows_hours)
        
        if lag_base_cols is None:
            self.lag_base_cols = [
                "temp", "humidity", "precip", "solarradiation", 
                "cloudcover", "precipcover",
            ]
        else:
            self.lag_base_cols = list(lag_base_cols)
    
    @staticmethod
    def load_hanoi_hourly_data(file_path: str) -> pd.DataFrame:
        """
        Load Hanoi hourly weather data from CSV file.
        Assumes the CSV has a 'datetime' column in YYYY-MM-DD HH:MM:SS format.
        """
        df = pd.read_csv(file_path)
        return df

    # ===================== GÓC NÂNG MẶT TRỜI =====================
    @staticmethod
    def _solar_elevation(index: pd.DatetimeIndex, lat: float) -> np.ndarray:
        doy = index.dayofyear.values
        hour = index.hour + index.minute / 60.0
        decl = 23.44 * np.sin(np.deg2rad(360 / 365.25 * (284 + doy))) 
        decl_rad = np.deg2rad(decl)
        lat_rad = np.deg2rad(lat)
        hour_angle = np.deg2rad((hour - 12) * 15)
        elev = np.arcsin(
            np.sin(lat_rad) * np.sin(decl_rad) +
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle)
        )
        return np.rad2deg(elev)

    # ===================== 1. FE THỜI GIAN & SOLAR =====================
    def _add_time_and_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Xử lý Index (Naive Datetime)
        if self.date_col in out.columns:
            out[self.date_col] = pd.to_datetime(out[self.date_col])
            out = out.set_index(self.date_col)
        else:
            out.index = pd.to_datetime(out.index)
            
        if out.index.tz is not None:
             out.index = out.index.tz_localize(None) # Đảm bảo Naive Datetime
             
        out = out.sort_index().asfreq("h")

        # Time features
        out["hour"] = out.index.hour
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
        out["dayofyear"] = out.index.dayofyear
        out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
        out["dayofweek"] = out.index.dayofweek
        out["is_weekend"] = out.index.dayofweek.isin([5, 6]).astype(int)
        
        # Seasonal features
        m = out.index.month
        out["season_spring"] = m.isin([3, 4]).astype(int)
        out["season_summer"] = m.isin([5, 6, 7, 8]).astype(int)
        out["season_autumn"] = m.isin([9, 10, 11]).astype(int)
        out["season_winter"] = 1 - (out["season_spring"] | out["season_summer"] | out["season_autumn"])
        
        # Solar Elevation (KHÔNG ROLLING)
        out["solar_elev"] = self._solar_elevation(out.index, self.lat)
        
        out = out.drop(columns=["hour", "dayofyear"], errors="ignore")
        return out

    # ===================== 2. FE DANH MỤC (ICON) =====================
    def _fe_categorical_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mã hóa One-Hot cho cột icon.
        """
        out = df.copy()
        
        if self.icon_col not in out.columns:
            return out # Bỏ qua nếu cột icon không tồn tại

        # Xử lý giá trị thiếu thành chuỗi 'missing'
        icon_col = out[self.icon_col].fillna("missing").astype(str)
        
        # Thực hiện One-Hot Encoding
        icon_dummies = pd.get_dummies(icon_col, prefix="icon", dtype=int)
        out = pd.concat([out, icon_dummies], axis=1)
        
        # Drop cột icon gốc
        out = out.drop(columns=[self.icon_col], errors="ignore")
        return out

    # ===================== 3. MONSOON & FE GIÓ =====================
    @staticmethod
    def _monsoon_zone(deg: float | int | None) -> str:
        if pd.isna(deg): return "Unknown"
        d = float(deg) % 360 
        if 20.0 <= d <= 80.0: return "NE"
        if 200.0 <= d <= 260.0: return "SW"
        return "Other"

    def _fe_wind_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        has_wind_data = self.winddir_col in out.columns and self.windsp_col in out.columns
        
        # Mặc định: Gió lặng (is_calm=1) và gió mùa "Other/Unknown" (monsoon_Other=1)
        # Nếu muốn tuân thủ strict requirement trước đó, hãy sử dụng logic có cờ missing
        # out["wind_data_missing"] = 0 
        
        if not has_wind_data:
            out["monsoon_NE"] = 0; out["monsoon_SW"] = 0; out["monsoon_Other"] = 1 # Hoặc 0 nếu dùng cờ missing
            out["winddir_sin"] = 0.0; out["winddir_cos"] = 0.0
            out["u_wind"] = 0.0; out["v_wind"] = 0.0
            out["is_calm"] = 1 # Hoặc 0 nếu dùng cờ missing
            return out

        spd = pd.to_numeric(out[self.windsp_col], errors="coerce").fillna(0.0)
        deg = pd.to_numeric(out[self.winddir_col], errors="coerce").fillna(0.0)
        
        # Monsoon
        out["monsoon"] = deg.apply(self._monsoon_zone).astype("category")
        out["monsoon_NE"] = (out["monsoon"] == "NE").astype(int)
        out["monsoon_SW"] = (out["monsoon"] == "SW").astype(int)
        out["monsoon_Other"] = (out["monsoon"] == "Other").astype(int)
        out = out.drop(columns=["monsoon"])

        # sin / cos và vector gió
        rad = np.deg2rad(deg % 360)
        out["winddir_sin"] = np.sin(rad)
        out["winddir_cos"] = np.cos(rad)

        out["u_wind"] = -spd * out["winddir_sin"]
        out["v_wind"] = -spd * out["winddir_cos"]

        # is_calm
        out["is_calm"] = (spd <= self.calm_thr).astype(int)

        return out

    # ===================== 4. FE MƯA =====================
    def _fe_precip_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "precip" not in out.columns:
            out["precip_flag"] = 0
            out["precip_24h_sum"] = 0.0
            out["hours_since_rain"] = 9999.0 
            return out

        precip_col = out["precip"].fillna(0)
        out["precip_flag"] = (precip_col > 0).astype(int)
        
        # Tổng mưa 24h trước
        out["precip_24h_sum"] = precip_col.shift(1).rolling(window=24, min_periods=1).sum().fillna(0.0)

        # Số giờ kể từ trận mưa cuối
        rain_mask = out["precip_flag"].astype(bool)
        last_rain_time = pd.Series(out.index, index=out.index).where(rain_mask).ffill()
        # Subtract two timestamp Series to obtain a Timedelta Series, then use .dt.total_seconds()
        hours_since_rain = (pd.Series(out.index, index=out.index) - last_rain_time).dt.total_seconds() / 3600.0
        out["hours_since_rain"] = hours_since_rain.fillna(9999.0)
        
        return out
        
    # ===================== 5. LAG & ROLLING (CHỈ CHO BASE COLS) =====================
    def _add_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        target_cols = [c for c in self.lag_base_cols if c in out.columns]

        for col in target_cols:
            # LAG (1, 3, 12, 24)
            for L in self.all_lags:
                out[f"{col}_lag{L}h"] = out[col].shift(L)

            # ROLLING (MEAN & STD) - Chỉ áp dụng cho Base Cols
            for w in self.roll_windows:
                base = out[col].rolling(window=w, min_periods=1)
                out[f"{col}_roll{w}h_mean"] = base.mean()
                out[f"{col}_roll{w}h_std"] = base.std().fillna(0.0)

        return out

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        # 1) Thời gian + Solar Elevation
        out = self._add_time_and_solar_features(out)
        
        # 2) Danh mục (ICON) 
        out = self._fe_categorical_block(out)
        
        # 3) Gió 
        out = self._fe_wind_block(out)

        # 4) Mưa 
        out = self._fe_precip_block(out)

        # 5) Lag & rolling cho các biến cơ sở (lag_base_cols)
        out = self._add_lag_rolling_features(out)

        # Dọn dẹp
        final_drop_cols = [
            self.winddir_col, self.windsp_col, 
            "dayofweek", "month",
        ]
        out = out.drop(columns=[c for c in final_drop_cols if c in out.columns], errors="ignore")
        
        return out

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature Engineering for Hanoi Hourly Weather Data')
    parser.add_argument('--input', type=str, required=False,
                        help='Input CSV file path (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, required=False,
                        help='Output CSV file path (optional, uses default if not provided)')
    
    args = parser.parse_args()
    
    # Set default paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_input = os.path.join(project_root, 'dataset', 'processed', 'Hanoi_Hourly_Selected.csv')
    default_output = os.path.join(project_root, 'dataset', 'processed', 'Hanoi_hourly_FE_full.csv')
    
    input_file = args.input if args.input else default_input
    output_file = args.output if args.output else default_output
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Apply Hourly Feature Engineering
    print("\nApplying Hourly Feature Engineering...")
    fe = HanoiHourlyFE(
        date_col="datetime",
        lat=21.0278,
        winddir_col="winddir",
        windsp_col="windspeed",
        icon_col="icon",
        calm_thr=0.5,
        all_lags_hours=(1, 3, 12, 24),
        rolling_windows_hours=(3, 6, 12, 24, 72, 168)
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
