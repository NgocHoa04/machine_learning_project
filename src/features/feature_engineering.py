# FE.py
import numpy as np
import pandas as pd
from typing import Sequence, Optional


class feature_engineering_class:
    """


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
    ) -> None:
        self.date_col = date_col
        self.sunrise_col = sunrise_col
        self.sunset_col = sunset_col
        self.winddir_col = winddir_col
        self.windsp_col = windsp_col
        self.lag_days = tuple(lag_days)
        self.roll_windows = tuple(roll_windows)
        self.calm_thr = calm_thr

        #các cột dùng để tạo lag/rolling
        if lag_base_cols is None:
            self.lag_base_cols = [
                "humidity",
                "windspeed",
                "precip",
                "solarradiation",
                "cloudcover",
                "precipcover",
                "daylength_hours",
            ]
        else:
            self.lag_base_cols = list(lag_base_cols)

    # ===================== 1. MONSOON & FE GIÓ =====================

    @staticmethod
    def _monsoon_zone(deg: float | int | None) -> str:
        """
        Phân vùng gió mùa cho Hà Nội:
        - NE  : 20–80°   (gió mùa Đông Bắc, đông-xuân, lạnh/ẩm)
        - SW  : 200–260° (gió mùa Tây Nam, hè, nóng/ẩm/giông)
        - Other / Unknown: còn lại hoặc giá trị thiếu
        """
        if pd.isna(deg):
            return "Unknown"

        d = float(deg) % 360  # đảm bảo 0–360

        if 20.0 <= d <= 80.0:
            return "NE"
        if 200.0 <= d <= 260.0:
            return "SW"
        return "Other"

    def _fe_wind_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FE gió từ winddir (+ windspeed nếu có):

        - monsoon (NE / SW / Other / Unknown) + one-hot:
            monsoon_NE, monsoon_SW, monsoon_Other
        - winddir_sin, winddir_cos (mã hoá góc chu kỳ)
        - u_wind, v_wind (vector gió, hướng 'from')
        - is_calm (gió lặng: speed <= calm_thr)

        Nếu thiếu cột winddir, tạo các cột 0/dummy để pipeline không vỡ.
        """
        out = df.copy()

        if self.winddir_col not in out.columns:
            # Không có hướng gió -> tạo dummy feature để giữ cấu trúc
            out["monsoon"] = "Unknown"
            out["monsoon"] = out["monsoon"].astype("category")
            out["monsoon_NE"] = 0
            out["monsoon_SW"] = 0
            out["monsoon_Other"] = 0
            out["winddir_sin"] = 0.0
            out["winddir_cos"] = 0.0
            out["u_wind"] = 0.0
            out["v_wind"] = 0.0
            out["is_calm"] = 0
            return out

        # 1) monsoon category + one-hot
        out["monsoon"] = out[self.winddir_col].apply(self._monsoon_zone).astype("category")
        out["monsoon_NE"] = (out["monsoon"] == "NE").astype(int)
        out["monsoon_SW"] = (out["monsoon"] == "SW").astype(int)
        out["monsoon_Other"] = (out["monsoon"] == "Other").astype(int)
        # Drop the original monsoon column
        out = out.drop(columns=["monsoon"])

        # 2) sin / cos của hướng gió
        rad = np.deg2rad(out[self.winddir_col].astype(float))
        out["winddir_sin"] = np.sin(rad)
        out["winddir_cos"] = np.cos(rad)

        # 3) vector gió u, v (dùng speed đã scale nếu có)
        if self.windsp_col in out.columns:
            spd = out[self.windsp_col].astype(float).fillna(0.0)
        else:
            spd = pd.Series(0.0, index=out.index)

        # winddir là hướng gió THỔI TỪ đâu đến:
        out["u_wind"] = -spd * np.sin(rad)
        out["v_wind"] = -spd * np.cos(rad)

        # 4) is_calm
        out["is_calm"] = (spd <= self.calm_thr).astype(int)

        for col in ["winddir_sin", "winddir_cos", "u_wind", "v_wind"]:
            out[col] = out[col].fillna(0.0)

        return out

    # ===================== 2. LAG & ROLLING (NO LEAKAGE) =====================

    def _add_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm lag & rolling cho nhiều biến thời tiết.
        - shift(1) trước rolling.
        - Aggregations: mean, std
        - Sort theo date_col
        """
        out = df.copy()
        out = out.sort_values(self.date_col)

        target_cols = [c for c in self.lag_base_cols if c in out.columns]

        for col in target_cols:
            # ---------- LAG ----------
            for L in self.lag_days:
                out[f"{col}_lag{L}"] = out[col].shift(L)

            # ---------- ROLLING ----------
            for w in self.roll_windows:
                # luôn shift(1) để tránh dùng dữ liệu ngày hiện tại (no leakage)
                base = out[col].rolling(window=w)
                out[f"{col}_roll{w}d_mean"] = base.mean()
                out[f"{col}_roll{w}d_std"] = base.std()

        return out

    # ===================== 3. FE THỜI GIAN + DAYLENGTH + MÙA =====================

    def _add_calendar_and_daylength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm các feature thời gian & độ dài ngày cho Hà Nội:

        - datetime -> month, dayofyear, dow
        - doy_sin, doy_cos (chu kỳ năm)
        - daylength_hours = (sunset - sunrise) / 3600
        - season_sin, season_cos (chu kỳ mùa trong năm)

        """
        out = df.copy()

        out[self.date_col] = pd.to_datetime(out[self.date_col])
        out["sunrise_dt"] = pd.to_datetime(out[self.sunrise_col])
        out["sunset_dt"] = pd.to_datetime(out[self.sunset_col])

        # Calendar basic
        out["month"] = out[self.date_col].dt.month
        out["dayofyear"] = out[self.date_col].dt.day_of_year
        out["dow"] = out[self.date_col].dt.dayofweek

        # Chu kỳ năm
        doy = out["dayofyear"]
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Độ dài ngày
        dl_seconds = (out["sunset_dt"] - out["sunrise_dt"]).dt.total_seconds()
        out["daylength_hours"] = dl_seconds / 3600.0

        # Chu kỳ mùa (theo day_of_year dựa trên sunrise_dt)
        out["day_of_year"] = out["sunrise_dt"].dt.dayofyear
        out["season_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
        out["season_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

        return out

    # ===================== 4. PUBLIC API – HÀM FE TỔNG HỢP =====================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 1) Thời gian + daylength + mùa
        out = self._add_calendar_and_daylength_features(out)

        # 2) Gió
        out = self._fe_wind_block(out)

        # 3) Lag & rolling
        out = self._add_lag_rolling_features(out)

        # Drop unnecessary columns
        for col in ["sunrise_dt", "sunset_dt", 'winddir', 'conditions', 'sunrise', 'sunset']:
            if col in out.columns:
                out = out.drop(columns=col)

        return out
