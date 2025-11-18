import datetime
import functools
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import plotly.graph_objects as go

from style import make_css

# ===================== 0. BACKGROUND (THEO TIME) =====================

TIME_BACKGROUND = {
    "morning": "https://images.unsplash.com/photo-1717099377307-142881f48cea?auto=format&fit=crop&w=1170&q=80",
    "afternoon": "https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=1600&q=80",
    "night": "https://images.unsplash.com/photo-1644579124055-87f8abea61f8?auto=format&fit=crop&w=1978&q=80",
}


def get_time_of_day(now=None) -> str:
    if now is None:
        now = datetime.datetime.now()
    h = now.hour
    if 5 <= h < 14:
        return "morning"
    elif 14 <= h < 19:
        return "afternoon"
    else:
        return "night"


CURRENT_TOD = get_time_of_day()
BG_URL = TIME_BACKGROUND.get(CURRENT_TOD, TIME_BACKGROUND["afternoon"])
CUSTOM_CSS = make_css(BG_URL)

# ===================== 1. PATH & LOAD DATA =====================


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset"

DAILY_RAW_PATH = DATASET_DIR / "raw" / "Hanoi Daily.csv"
DAILY_FE_PATH = DATASET_DIR / "processed" / "Hanoi_daily_FE_full.csv"

HOURLY_RAW_PATH = DATASET_DIR / "raw" / "Hanoi Hourly.csv"
HOURLY_FE_PATH = DATASET_DIR / "processed" / "Hanoi_hourly_FE_full.csv"

MODELS_DIR = PROJECT_ROOT / "src" / "config" / "models_pkl"

# ===== DAILY RAW / FE =====
if not DAILY_RAW_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file daily raw: {DAILY_RAW_PATH}")

if not DAILY_FE_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file daily FE: {DAILY_FE_PATH}")

HANOI_DAILY_RAW = pd.read_csv(DAILY_RAW_PATH)
HANOI_DAILY_RAW["datetime"] = pd.to_datetime(HANOI_DAILY_RAW["datetime"])
HANOI_DAILY_RAW = HANOI_DAILY_RAW.sort_values("datetime")
HANOI_TEMP_DAILY = HANOI_DAILY_RAW[["datetime", "temp"]].copy()

HANOI_FE = pd.read_csv(DAILY_FE_PATH)
if "datetime" in HANOI_FE.columns:
    HANOI_FE["datetime"] = pd.to_datetime(HANOI_FE["datetime"])
    HANOI_FE = HANOI_FE.sort_values("datetime")

FIRST_DAILY_DATE = HANOI_TEMP_DAILY["datetime"].min().date()
LAST_DAILY_DATE = HANOI_TEMP_DAILY["datetime"].max().date()

# ===== HOURLY RAW / FE =====
if not HOURLY_RAW_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file hourly raw: {HOURLY_RAW_PATH}")

HANOI_HOURLY_RAW = pd.read_csv(HOURLY_RAW_PATH)
HANOI_HOURLY_RAW["datetime"] = pd.to_datetime(HANOI_HOURLY_RAW["datetime"])
HANOI_HOURLY_RAW = HANOI_HOURLY_RAW.sort_values("datetime")
HANOI_TEMP_HOURLY = HANOI_HOURLY_RAW[["datetime", "temp"]].copy()

FIRST_HOURLY_DT = HANOI_TEMP_HOURLY["datetime"].min()
LAST_HOURLY_DT = HANOI_TEMP_HOURLY["datetime"].max()

if not HOURLY_FE_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file hourly FE: {HOURLY_FE_PATH}")

HANOI_HOURLY_FE = pd.read_csv(HOURLY_FE_PATH)
if "datetime" in HANOI_HOURLY_FE.columns:
    HANOI_HOURLY_FE["datetime"] = pd.to_datetime(HANOI_HOURLY_FE["datetime"])
    HANOI_HOURLY_FE = HANOI_HOURLY_FE.sort_values("datetime")


# ===================== 2. UTILS =====================

def latest_observed_temp_daily():
    row = HANOI_TEMP_DAILY.iloc[-1]
    return row["datetime"].date(), float(row["temp"])


def latest_observed_temp_hourly():
    row = HANOI_TEMP_HOURLY.iloc[-1]
    return row["datetime"], float(row["temp"])


def _to_date(d) -> datetime.date:
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    s = str(d).strip()
    if not s or s.lower().startswith("now"):
        return datetime.date.today()
    s = s.replace("/", "-")
    return datetime.date.fromisoformat(s[:10])


# ===================== 3. LOAD MODELS (DAILY & HOURLY) =====================

def _daily_model_path(h: int) -> Path:
    return MODELS_DIR / f"hanoi_temp_v1_h{h}.pkl"


# chỉnh prefix này theo tên 5 file hourly .pkl của bạn
HOURLY_MODEL_PREFIX = "hanoi_temp_v1_20251117_154939_h"


def _hourly_model_path(h: int) -> Path:
    return MODELS_DIR / f"{HOURLY_MODEL_PREFIX}{h}.pkl"


@functools.lru_cache(maxsize=8)
def load_daily_model(h: int):
    path = _daily_model_path(h)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file DAILY model cho horizon h{h}: {path}")
    model = joblib.load(path)
    return model


@functools.lru_cache(maxsize=8)
def load_hourly_model(h: int):
    path = _hourly_model_path(h)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file HOURLY model cho horizon h{h}: {path}")
    model = joblib.load(path)
    return model


def build_daily_features_for_model(base_date: datetime.date) -> pd.DataFrame:
    """
    Lấy dòng FE gần nhất <= base_date làm input cho model daily.
    """
    if "datetime" in HANOI_FE.columns:
        mask = HANOI_FE["datetime"].dt.date <= base_date
        sub = HANOI_FE.loc[mask]
        if sub.empty:
            row = HANOI_FE.iloc[[-1]].copy()
        else:
            row = sub.iloc[[-1]].copy()
    else:
        row = HANOI_FE.iloc[[-1]].copy()

    if "datetime" in row.columns:
        row = row.drop(columns=["datetime"])
    return row


def build_hourly_features_for_model(base_dt: datetime.datetime) -> pd.DataFrame:
    """
    Lấy dòng FE gần nhất <= base_dt làm input cho model hourly
    và căn chỉnh đúng feature_names.
    """
    if "datetime" in HANOI_HOURLY_FE.columns:
        mask = HANOI_HOURLY_FE["datetime"] <= base_dt
        sub = HANOI_HOURLY_FE.loc[mask]
        if sub.empty:
            base_row = HANOI_HOURLY_FE.iloc[[-1]].copy()
        else:
            base_row = sub.iloc[[-1]].copy()
    else:
        base_row = HANOI_HOURLY_FE.iloc[[-1]].copy()

    drop_cols = [c for c in ["datetime", "temp"] if c in base_row.columns]
    if drop_cols:
        base_row = base_row.drop(columns=drop_cols)

    model = load_hourly_model(1)
    booster = model.get_booster()
    feat_names = booster.feature_names

    if feat_names is not None:
        missing = set(feat_names) - set(base_row.columns)
        extra = set(base_row.columns) - set(feat_names)
        if missing:
            print("WARNING - hourly FE thiếu cột:", missing)
        if extra:
            print("NOTE - hourly FE thừa cột (bỏ đi khi predict):", extra)

        base_row = base_row[feat_names]

    return base_row


# ===================== 4. DAILY FORECAST LOGIC (CÓ target_date) =====================

def predict_temperature_daily(horizon: int, target_date_str: str):
    """
    Daily forecast với horizon H (1..5), base tại target_date:
    - target_date nằm trong [FIRST_DAILY_DATE, LAST_DAILY_DATE]
    - nếu rỗng / sai format -> dùng LAST_DAILY_DATE
    """
    horizon = int(horizon)
    horizon = max(1, min(horizon, 5))

    if target_date_str is None or str(target_date_str).strip() == "":
        base_date = LAST_DAILY_DATE
    else:
        try:
            base_date = _to_date(target_date_str)
        except Exception:
            base_date = LAST_DAILY_DATE

    if base_date < FIRST_DAILY_DATE:
        base_date = FIRST_DAILY_DATE
    if base_date > LAST_DAILY_DATE:
        base_date = LAST_DAILY_DATE

    mask_daily = HANOI_TEMP_DAILY["datetime"].dt.date <= base_date
    base_row = HANOI_TEMP_DAILY.loc[mask_daily].iloc[-1]
    base_date = base_row["datetime"].date()
    base_temp = float(base_row["temp"])

    X_row = build_daily_features_for_model(base_date)

    results = []
    for h in range(1, horizon + 1):
        forecast_date = base_date + datetime.timedelta(days=h)
        model = load_daily_model(h)
        y_pred = model.predict(X_row)[0]
        results.append(
            {"date": forecast_date.isoformat(), "temp": float(y_pred), "horizon": h}
        )

    return base_date, base_temp, results, LAST_DAILY_DATE


def get_actual_daily_for_plot(
    base_date: datetime.date,
    horizon: int,
    last_data_date: datetime.date,
    history_days: int = 7,
):
    end = min(base_date + datetime.timedelta(days=horizon), last_data_date)
    start = base_date - datetime.timedelta(days=history_days - 1)

    mask = (
        (HANOI_TEMP_DAILY["datetime"].dt.date >= start)
        & (HANOI_TEMP_DAILY["datetime"].dt.date <= end)
    )
    sub = HANOI_TEMP_DAILY.loc[mask].copy()

    results = []
    for _, row in sub.iterrows():
        d = row["datetime"].date().isoformat()
        results.append({"date": d, "temp": float(row["temp"])})
    return results


# ===================== 5. HOURLY FORECAST + PLOT (CÓ date & hour) =====================

def get_hourly_actual_for_plot(
    base_dt: datetime.datetime,
    horizon_hours: int,
    last_data_dt: datetime.datetime,
    history_hours: int = 24,
):
    end = min(base_dt + datetime.timedelta(hours=horizon_hours), last_data_dt)
    start = base_dt - datetime.timedelta(hours=history_hours - 1)

    mask = (
        (HANOI_TEMP_HOURLY["datetime"] >= start)
        & (HANOI_TEMP_HOURLY["datetime"] <= end)
    )
    sub = HANOI_TEMP_HOURLY.loc[mask].copy()

    results = []
    for _, row in sub.iterrows():
        results.append(
            {"datetime": row["datetime"], "temp": float(row["temp"])}
        )
    return results


def predict_temperature_hourly_next5(target_date_str: str, target_hour: int):
    """
    Forecast hourly với base = (target_date, target_hour).
    - target_date rỗng -> dùng LAST_HOURLY_DT.date()
    - sau khi combine -> snap về mốc gần nhất <= trong data hourly.
    """
    # xác định date
    if target_date_str is None or str(target_date_str).strip() == "":
        base_date = LAST_HOURLY_DT.date()
    else:
        try:
            base_date = _to_date(target_date_str)
        except Exception:
            base_date = LAST_HOURLY_DT.date()

    # clamp hour
    try:
        h_int = int(target_hour)
    except Exception:
        h_int = LAST_HOURLY_DT.hour
    h_int = max(0, min(23, h_int))

    desired_dt = datetime.datetime.combine(base_date, datetime.time(hour=h_int))

    # snap về timestamp có thật trong data (gần nhất <= desired_dt)
    mask = HANOI_TEMP_HOURLY["datetime"] <= desired_dt
    if mask.any():
        base_row = HANOI_TEMP_HOURLY.loc[mask].iloc[-1]
    else:
        base_row = HANOI_TEMP_HOURLY.iloc[0]

    base_dt = base_row["datetime"]
    base_temp = float(base_row["temp"])

    X_row = build_hourly_features_for_model(base_dt)

    forecast_list = []
    box_texts = []

    for h in range(1, 6):
        model = load_hourly_model(h)
        y_pred = model.predict(X_row)[0]
        temp_val = float(y_pred)

        t = base_dt + datetime.timedelta(hours=h)
        forecast_list.append({"datetime": t, "temp": temp_val, "horizon": h})

        time_label = t.strftime("%H:%M<br>%d-%m")
        text = (
            f"<div class='mini-header'>H+{h}</div>"
            f"<div class='mini-temp'>{temp_val:.1f}°C</div>"
            f"<div class='mini-sub'>{time_label}</div>"
        )
        box_texts.append(text)

    while len(box_texts) < 5:
        box_texts.append("")

    return base_dt, base_temp, forecast_list, box_texts, LAST_HOURLY_DT


# ===================== 6. MAIN FORECAST FN FOR UI =====================

def run_forecast(horizon, target_date_daily, hourly_date, hourly_hour):
    horizon = int(horizon)

    # ----- DAILY -----
    base_date, base_temp, forecast_daily, last_data_date = predict_temperature_daily(
        horizon, target_date_daily
    )
    actual_daily = get_actual_daily_for_plot(
        base_date, horizon, last_data_date, history_days=7
    )

    today_str = datetime.date.today().isoformat()

    summary_lines = []
    summary_lines.append("#### Hanoi Temperature Forecast")
    summary_lines.append(f"- **Run date:** {today_str}")
    summary_lines.append(
        f"- **Daily base date (t0):** {base_date.isoformat()} with actual {base_temp:.1f}°C"
    )
    summary_lines.append(f"- **Daily horizon:** {len(forecast_daily)} day(s) ahead")
    summary_lines.append(f"- **Model:** XGBoost\n")

    if forecast_daily:
        first = forecast_daily[0]
        summary_lines.append(
            f"**Day 1 forecast ({first['date']}): {first['temp']:.1f}°C**"
        )

    summary_md = "\n".join(summary_lines)

    # DAILY TABLE
    daily_rows = []
    for r in actual_daily:
        t = round(r["temp"], 1)
        daily_rows.append([r["date"], t, "actual"])
    for r in forecast_daily:
        t = round(r["temp"], 1)
        daily_rows.append([r["date"], t, "forecast"])

    df_daily = pd.DataFrame(daily_rows, columns=["Date", "Temp (°C)", "Type"])

    # DAILY PLOT
    fig_daily = go.Figure()

    if actual_daily:
        dates_a = [r["date"] for r in actual_daily]
        temps_a = [round(r["temp"], 1) for r in actual_daily]
        fig_daily.add_trace(
            go.Scatter(
                x=dates_a,
                y=temps_a,
                mode="lines+markers",
                name="actual",
                line_shape="spline",
                line=dict(color="#4FC3F7", width=3),
                marker=dict(size=7),
            )
        )

    if forecast_daily:
        dates_f = [r["date"] for r in forecast_daily]
        temps_f = [round(r["temp"], 1) for r in forecast_daily]
        fig_daily.add_trace(
            go.Scatter(
                x=dates_f,
                y=temps_f,
                mode="lines+markers",
                name="forecast",
                line_shape="spline",
                line=dict(color="#FFB74D", width=3, dash="dash"),
                marker=dict(size=7),
            )
        )

    fig_daily.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Date",
        yaxis_title="Temp (°C)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECEFF1"),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.65)",
            bordercolor="rgba(148,163,184,0.4)",
            borderwidth=1,
        ),
    )

    # ----- HOURLY (base date + hour) -----
    base_dt_hour, base_temp_hour, hourly_forecast, hour_boxes, last_hourly_dt = (
        predict_temperature_hourly_next5(hourly_date, hourly_hour)
    )
    hourly_actual = get_hourly_actual_for_plot(
        base_dt_hour, horizon_hours=5, last_data_dt=last_hourly_dt, history_hours=24
    )

    fig_hourly = go.Figure()

    if hourly_actual:
        times_a = [r["datetime"] for r in hourly_actual]
        temps_a = [round(r["temp"], 1) for r in hourly_actual]
        fig_hourly.add_trace(
            go.Scatter(
                x=times_a,
                y=temps_a,
                mode="lines+markers",
                name="actual (hourly)",
                line_shape="spline",
                line=dict(color="#4FC3F7", width=2.5),
                marker=dict(size=5),
            )
        )

    if hourly_forecast:
        times_f = [r["datetime"] for r in hourly_forecast]
        temps_f = [round(r["temp"], 1) for r in hourly_forecast]
        fig_hourly.add_trace(
            go.Scatter(
                x=times_f,
                y=temps_f,
                mode="lines+markers",
                name="forecast (next 5h)",
                line_shape="spline",
                line=dict(color="#FFB74D", width=2.5, dash="dash"),
                marker=dict(size=6),
            )
        )

    fig_hourly.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Datetime",
        yaxis_title="Temp (°C)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECEFF1"),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.65)",
            bordercolor="rgba(148,163,184,0.4)",
            borderwidth=1,
        ),
        title="Hourly: last 24h + next 5h",
    )

    return (
        summary_md,
        df_daily,
        fig_daily,
        fig_hourly,
    )


# ===================== 7. HISTORICAL DATA (DAILY) =====================

def get_historical_data(start_date, end_date):
    try:
        start = _to_date(start_date)
        end = _to_date(end_date)
        if start > end:
            start, end = end, start
    except Exception:
        end = LAST_DAILY_DATE
        start = end - datetime.timedelta(days=30)

    mask = (
        (HANOI_TEMP_DAILY["datetime"].dt.date >= start)
        & (HANOI_TEMP_DAILY["datetime"].dt.date <= end)
    )
    sub = HANOI_TEMP_DAILY.loc[mask].copy()

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected range",
            xaxis_title="Date",
            yaxis_title="Temp (°C)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECEFF1"),
        )
        avg_text = "**Average:** N/A"
        max_text = "**Maximum:** N/A"
        min_text = "**Minimum:** N/A"
        return fig, avg_text, max_text, min_text

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["datetime"],
            y=sub["temp"],
            mode="lines",
            name="historical temp",
            line=dict(color="#4FC3F7", width=2.5),
        )
    )
    fig.update_layout(
        title="Historical Temperature",
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECEFF1"),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
    )

    avg_val = sub["temp"].mean()
    max_val = sub["temp"].max()
    min_val = sub["temp"].min()

    avg_text = f"**Average:** {avg_val:.1f}°C"
    max_text = f"**Maximum:** {max_val:.1f}°C"
    min_text = f"**Minimum:** {min_val:.1f}°C"

    return fig, avg_text, max_text, min_text


# ===================== 8. GRADIO UI =====================

with gr.Blocks(css=CUSTOM_CSS, title="Hanoi Temperature Forecast") as demo:

    with gr.Column(elem_classes="app-container"):
        gr.Markdown("## Hanoi Temperature Forecast", elem_classes="main-title")
        gr.Markdown(
            "Glass-style dashboard for short-term **daily & hourly temperature forecast** in Hanoi "
            "using horizon-specific XGBoost models.",
            elem_classes="subtext",
        )

        last_date, last_temp = latest_observed_temp_daily()

        with gr.Row():
            # LEFT: Location + last observed
            with gr.Column(scale=1, elem_classes="glass-card"):
                gr.Markdown("#### Hanoi, Viet Nam")

                gr.Markdown(
                    f"<div class='label-small'>Last observed (daily)</div>"
                    f"<div class='metric-value'>{last_temp:.1f}°C</div>"
                    f"<div class='subtext'>on {last_date.isoformat()}</div>",
                )

                gr.Markdown("#### Model Info")
                gr.Markdown(
                    "- 5 separate **daily** XGBoost models used for H1–H5.\n"
                    "- 5 **hourly** XGBoost models used for the next 5 hours.\n"
                    "- Input: last available feature vectors from FE pipelines.",
                    elem_classes="subtext",
                )

            # RIGHT: Controls + summary
            with gr.Column(scale=2, elem_classes="glass-card"):
                gr.Markdown("#### Forecast settings")

                # Daily settings
                with gr.Row():
                    horizon = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Daily horizon (days ahead, max 5)",
                    )
                    target_date_daily = gr.Textbox(
                        label="Daily base date (YYYY-MM-DD)",
                        value=LAST_DAILY_DATE.isoformat(),
                    )

                # Hourly settings
                with gr.Row():
                    hourly_date = gr.Textbox(
                        label="Hourly base date (YYYY-MM-DD)",
                        value=LAST_HOURLY_DT.date().isoformat(),
                    )
                    hourly_hour = gr.Slider(
                        minimum=0,
                        maximum=23,
                        value=int(LAST_HOURLY_DT.hour),
                        step=1,
                        label="Hourly base hour (0–23)",
                    )

                forecast_btn = gr.Button("Run forecast", variant="primary")
                summary_box = gr.Markdown()

        # DAILY PLOT
        with gr.Row():
            with gr.Column(elem_classes="glass-card-plot"):
                gr.Markdown("#### Daily: actual vs forecast")
                daily_plot = gr.Plot(show_label=False)

        # HOURLY PLOT
        with gr.Row():
            with gr.Column(elem_classes="glass-card-plot"):
                gr.Markdown("#### Hourly: last 24h + next 5h")
                hourly_plot = gr.Plot(show_label=False)

        # DAILY TABLE
        with gr.Row():
            with gr.Column(elem_classes=["glass-table-card"]):
                gr.Markdown("#### Daily details")
                daily_table = gr.Dataframe(
                    headers=["Date", "Temp (°C)", "Type"],
                    row_count=(0, "dynamic"),
                    col_count=(3, "fixed"),
                    interactive=False,
                    label="",
                )

        # HISTORICAL SECTION
        default_hist_end = LAST_DAILY_DATE
        default_hist_start = LAST_DAILY_DATE - datetime.timedelta(days=90)

        with gr.Row():
            with gr.Column(elem_classes="glass-card-plot"):
                gr.Markdown("### Historical Data (Daily)")

                with gr.Row():
                    hist_start = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value=default_hist_start.isoformat(),
                    )
                    hist_end = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=default_hist_end.isoformat(),
                    )

                hist_btn = gr.Button("Show historical data")

                hist_plot = gr.Plot(show_label=False)

                with gr.Row():
                    hist_avg = gr.Markdown()
                    hist_max = gr.Markdown()
                    hist_min = gr.Markdown()

        # ========= WIRES =========
        forecast_btn.click(
            fn=run_forecast,
            inputs=[horizon, target_date_daily, hourly_date, hourly_hour],
            outputs=[
                summary_box,
                daily_table,
                daily_plot,
                hourly_plot,
            ],
        )

        hist_btn.click(
            fn=get_historical_data,
            inputs=[hist_start, hist_end],
            outputs=[hist_plot, hist_avg, hist_max, hist_min],
        )

if __name__ == "__main__":
    demo.launch()
