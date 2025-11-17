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

# app.py: project_root/src/app/app.py → project_root = parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
DAILY_RAW_PATH = DATASET_DIR / "raw" / "Hanoi Daily.csv"          # có datetime + temp
DAILY_FE_PATH = DATASET_DIR / "processed" / "Hanoi_daily_FE_full.csv"  # FE, có datetime + các feature

MODELS_DIR = PROJECT_ROOT / "src" / "config" / "models_pkl"  # chứa các file hanoi_temp_v1_h{h}.pkl

if not DAILY_RAW_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file daily raw: {DAILY_RAW_PATH}")

if not DAILY_FE_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file daily FE: {DAILY_FE_PATH}")

# Raw daily: dùng cho lịch sử + last observed
HANOI_DAILY_RAW = pd.read_csv(DAILY_RAW_PATH)
HANOI_DAILY_RAW["datetime"] = pd.to_datetime(HANOI_DAILY_RAW["datetime"])
HANOI_DAILY_RAW = HANOI_DAILY_RAW.sort_values("datetime")
HANOI_TEMP_DAILY = HANOI_DAILY_RAW[["datetime", "temp"]].copy()

# FE daily: dùng làm input feature cho model .pkl
HANOI_FE = pd.read_csv(DAILY_FE_PATH)
if "datetime" in HANOI_FE.columns:
    HANOI_FE["datetime"] = pd.to_datetime(HANOI_FE["datetime"])
    HANOI_FE = HANOI_FE.sort_values("datetime")


# ===================== 2. UTILS =====================

def latest_observed_temp_daily():
    row = HANOI_TEMP_DAILY.iloc[-1]
    return row["datetime"].date(), float(row["temp"])


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


# ===================== 3. LOAD MODELS H1..H5 =====================

def _model_path(h: int) -> Path:
    # Ví dụ: config/models_pkl/hanoi_temp_v1_h1.pkl
    return MODELS_DIR / f"hanoi_temp_v1_h{h}.pkl"


@functools.lru_cache(maxsize=8)
def load_daily_model(h: int):
    path = _model_path(h)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file model cho horizon h{h}: {path}")
    model = joblib.load(path)
    return model


def build_features_for_model(horizon: int) -> pd.DataFrame:
    """
    Lấy 1 dòng feature mới nhất từ file FE để đưa vào model .pkl.
    Giờ CSV FE và model đã trùng feature_names, nên chỉ cần:
    - lấy dòng cuối cùng
    - drop 'datetime' (model không dùng)
    """
    base_row = HANOI_FE.iloc[[-1]].copy()

    if "datetime" in base_row.columns:
        base_row = base_row.drop(columns=["datetime"])

    return base_row


# ===================== 4. DAILY FORECAST LOGIC =====================

def predict_temperature_daily(horizon: int, model_name: str):
    """
    Dự báo daily cho H ngày tới (tối đa 5) dựa trên 5 model XGBoost:
    - H1: model dự đoán ngày thứ 1
    - H2: model dự đoán ngày thứ 2
    ...
    - H5: model dự đoán ngày thứ 5

    Horizon input (1..5). Dự báo start từ NGÀY CUỐI CÙNG có trong lịch sử.
    """
    horizon = int(horizon)
    horizon = max(1, min(horizon, 5))

    last_date, last_temp = latest_observed_temp_daily()

    results = []
    for h in range(1, horizon + 1):
        forecast_date = last_date + datetime.timedelta(days=h)

        # 1) Lấy 1 dòng feature từ FE (DataFrame 1 x N, cột trùng với model)
        X_row = build_features_for_model(h)

        # 2) Load đúng model cho horizon
        model = load_daily_model(h)

        # 3) Predict trực tiếp với DataFrame (XGBoost dùng tên cột để map)
        y_pred = model.predict(X_row)[0]

        results.append(
            {
                "date": forecast_date.isoformat(),
                "temp": float(y_pred),
                "horizon": h,
            }
        )

    return last_date, last_temp, results


def get_actual_daily_for_plot(last_date: datetime.date, history_days: int = 7):
    start = last_date - datetime.timedelta(days=history_days - 1)
    mask = (
        (HANOI_TEMP_DAILY["datetime"].dt.date >= start)
        & (HANOI_TEMP_DAILY["datetime"].dt.date <= last_date)
    )
    sub = HANOI_TEMP_DAILY.loc[mask].copy()

    results = []
    for _, row in sub.iterrows():
        d = row["datetime"].date().isoformat()
        results.append({"date": d, "temp": float(row["temp"])})
    return results


# ===================== 5. MAIN FORECAST FN FOR UI =====================

def run_forecast(horizon, model_name):
    horizon = int(horizon)
    last_date, last_temp, forecast_daily = predict_temperature_daily(horizon, model_name)
    actual_daily = get_actual_daily_for_plot(last_date, history_days=7)

    today_str = datetime.date.today().isoformat()

    # ===== SUMMARY TEXT =====
    summary_lines = []
    summary_lines.append("#### Hanoi Temperature Forecast")
    summary_lines.append(f"- **Run date:** {today_str}")
    summary_lines.append(f"- **Last observed (daily):** {last_temp:.1f}°C on {last_date.isoformat()}")
    summary_lines.append(f"- **Horizon:** {len(forecast_daily)} day(s) ahead")
    summary_lines.append(f"- **Model:** {model_name}\n")

    if forecast_daily:
        first = forecast_daily[0]
        summary_lines.append(
            f"**Day 1 forecast ({first['date']}): {first['temp']:.1f}°C**"
        )

    summary_md = "\n".join(summary_lines)

    # ===== DAILY TABLE =====
    daily_rows = []
    for r in actual_daily:
        daily_rows.append([r["date"], r["temp"], "actual"])
    for r in forecast_daily:
        daily_rows.append([r["date"], r["temp"], "forecast"])

    df_daily = pd.DataFrame(daily_rows, columns=["Date", "Temp (°C)", "Type"])

    # ===== DAILY PLOT =====
    fig_daily = go.Figure()

    if actual_daily:
        dates_a = [r["date"] for r in actual_daily]
        temps_a = [r["temp"] for r in actual_daily]
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
        temps_f = [r["temp"] for r in forecast_daily]
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

    return summary_md, df_daily, fig_daily


# ===================== 6. HISTORICAL DATA =====================

def get_historical_data(start_date, end_date):
    try:
        start = _to_date(start_date)
        end = _to_date(end_date)
        if start > end:
            start, end = end, start
    except Exception:
        end = HANOI_TEMP_DAILY["datetime"].max().date()
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


# ===================== 7. GRADIO UI =====================

with gr.Blocks(css=CUSTOM_CSS, title="Hanoi Temperature Forecast") as demo:

    with gr.Column(elem_classes="app-container"):
        gr.Markdown("## Hanoi Temperature Forecast", elem_classes="main-title")
        gr.Markdown(
            "Glass-style dashboard for short-term **daily temperature forecast** in Hanoi "
            "using horizon-specific XGBoost models (H1–H5).",
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
                    "- 5 separate XGBoost models used for H1–H5.\n"
                    "- Input: last available feature vector from FE pipeline.\n"
                    "- Output: mean daily temperature for each of the next days.",
                    elem_classes="subtext",
                )

            # RIGHT: Controls + summary
            with gr.Column(scale=2, elem_classes="glass-card"):
                gr.Markdown("#### Forecast settings")

                with gr.Row():
                    horizon = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Horizon (days ahead, max 5)",
                    )

                    model_name = gr.Dropdown(
                        choices=["xgboost_v1"],
                        value="xgboost_v1",
                        label="Model",
                    )

                forecast_btn = gr.Button("Run forecast", variant="primary")
                summary_box = gr.Markdown()

        # DAILY PLOT
        with gr.Row():
            with gr.Column(elem_classes="glass-card-plot"):
                gr.Markdown("#### Daily: actual vs forecast")
                daily_plot = gr.Plot(show_label=False)

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
        last_daily_date = HANOI_TEMP_DAILY["datetime"].max().date()
        default_hist_end = last_daily_date
        default_hist_start = last_daily_date - datetime.timedelta(days=90)

        with gr.Row():
            with gr.Column(elem_classes="glass-card-plot"):
                gr.Markdown("### 📊 Historical Data")

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
            inputs=[horizon, model_name],
            outputs=[summary_box, daily_table, daily_plot],
        )

        hist_btn.click(
            fn=get_historical_data,
            inputs=[hist_start, hist_end],
            outputs=[hist_plot, hist_avg, hist_max, hist_min],
        )

if __name__ == "__main__":
    demo.launch()
