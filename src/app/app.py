import datetime
import random

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from style import make_css  

# ===================== 0. BACKGROUND (THEO TIME) =====================

TIME_BACKGROUND = {
    "morning": "https://images.unsplash.com/photo-1717099377307-142881f48cea?auto=format&fit=crop&w=1170&q=80",
    "afternoon": "https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=1600&q=80",
    "night": "https://images.unsplash.com/photo-1644579124055-87f8abea61f8?auto=format&fit=crop&w=1978&q=80",
}


def get_time_of_day(now=None):
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

# ===================== 1. LOAD DATA (DAILY & HOURLY) =====================

# Daily
HANOI_DF = pd.read_csv("dataset/raw/Hanoi Daily.csv")
HANOI_DF["datetime"] = pd.to_datetime(HANOI_DF["datetime"])
HANOI_DF = HANOI_DF.sort_values("datetime")
HANOI_TEMP_DAILY = HANOI_DF[["datetime", "temp"]].copy()

# Hourly
HANOI_HOURLY_DF = pd.read_csv("dataset/raw/Hanoi Hourly.csv")
HANOI_HOURLY_DF["datetime"] = pd.to_datetime(HANOI_HOURLY_DF["datetime"])
HANOI_HOURLY_DF = HANOI_HOURLY_DF.sort_values("datetime")
HANOI_TEMP_HOURLY = HANOI_HOURLY_DF[["datetime", "temp"]].copy()


def latest_observed_temp_daily():
    row = HANOI_TEMP_DAILY.iloc[-1]
    return row["datetime"].date().isoformat(), float(row["temp"])


def latest_observed_temp_hourly():
    row = HANOI_TEMP_HOURLY.iloc[-1]
    return row["datetime"], float(row["temp"])


# ===================== 2. DUMMY MODELS (THAY BẰNG MODEL THẬT SAU) =====================

def predict_temperature_daily(target_date: str, horizon: int, model_name: str):
    """
    Daily forecast demo (dummy). Sau này thay bằng model daily thật.
    """
    start_date = datetime.date.fromisoformat(target_date)
    base_temp = 28
    results = []

    for i in range(horizon):
        d = start_date + datetime.timedelta(days=i)
        temp = base_temp + random.uniform(-3, 3)
        results.append({"date": d.isoformat(), "temp": round(temp, 2)})

    return results


def predict_temperature_next_hour(current_temp: float, model_name: str):
    """
    Hourly one-step-ahead demo (dummy).
    Dự báo temp giờ tiếp theo = temp hiện tại + nhiễu nhỏ.
    """
    return round(current_temp + random.uniform(-1.5, 1.5), 2)


# ===================== 3. LẤY ACTUAL =====================

def get_actual_daily(target_date: str, history_days: int = 7):
    target = datetime.date.fromisoformat(target_date)
    start = target - datetime.timedelta(days=history_days)

    mask = (
        (HANOI_TEMP_DAILY["datetime"].dt.date >= start)
        & (HANOI_TEMP_DAILY["datetime"].dt.date < target)
    )

    sub = HANOI_TEMP_DAILY.loc[mask].copy()
    results = []

    for _, row in sub.iterrows():
        d = row["datetime"].date().isoformat()
        results.append({"date": d, "temp": float(row["temp"])})

    return results


# ===================== 4. HÀM CHÍNH DÙNG CHO UI =====================

def run_forecast(target_date, horizon, model_name):
    horizon = int(horizon)

    # ---- DAILY: actual history + forecast horizon ----
    actual_daily = get_actual_daily(target_date, history_days=7)
    forecast_daily = predict_temperature_daily(target_date, horizon, model_name)

    # ---- HOURLY: current hour + forecast next hour ----
    current_dt, current_temp = latest_observed_temp_hourly()
    next_dt = current_dt + datetime.timedelta(hours=1)
    next_temp = predict_temperature_next_hour(current_temp, model_name)

    # ===== 1) SUMMARY TEXT (Markdown) =====
    today_str = datetime.date.today().isoformat()
    summary_lines = []

    summary_lines.append("#### Hanoi Temperature Forecast")
    summary_lines.append(f"- **Run date:** {today_str}")
    summary_lines.append(f"- **Start date (daily):** {target_date}")
    summary_lines.append(f"- **Horizon (daily):** {horizon} day(s)")
    summary_lines.append(f"- **Model:** {model_name}\n")

    if forecast_daily:
        first = forecast_daily[0]
        summary_lines.append(
            f"**Day 1 daily forecast ({first['date']}): {first['temp']:.1f}°C (mean)**"
        )

    # hourly highlight
    summary_lines.append("\n**Hourly snapshot (based on latest record in dataset):**")
    summary_lines.append(
        f"- Current hour: {current_dt.strftime('%Y-%m-%d %H:%M')}, "
        f"actual temp: **{current_temp:.1f}°C**"
    )
    summary_lines.append(
        f"- Next hour: {next_dt.strftime('%Y-%m-%d %H:%M')}, "
        f"forecast temp: **{next_temp:.1f}°C**"
    )

    summary_md = "\n".join(summary_lines)

    # ===== 2) DAILY TABLE =====
    daily_rows = []
    for r in actual_daily:
        daily_rows.append([r["date"], r["temp"], "actual"])
    for r in forecast_daily:
        daily_rows.append([r["date"], r["temp"], "forecast"])

    df_daily = pd.DataFrame(daily_rows, columns=["Date", "Temp (°C)", "Type"])

    # ===== 3) DAILY PLOT (Plotly – đường cong mềm) =====
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

    # ===== 4) HOURLY TABLE (current vs next hour) =====
    df_hourly = pd.DataFrame(
        [
            [current_dt.strftime("%Y-%m-%d %H:%M"), current_temp, "actual"],
            [next_dt.strftime("%Y-%m-%d %H:%M"), next_temp, "forecast"],
        ],
        columns=["Datetime", "Temp (°C)", "Type"],
    )

    # ===== 5) HOURLY BOX (CHỈ HIỆN NHIỆT ĐỘ GIỜ TIẾP THEO) =====
    hourly_box_html = f"""
    <div class="next-hour-temp-box">
        <span class="next-hour-temp-value">{next_temp:.1f}°C</span>
    </div>
    """

    return summary_md, df_daily, fig_daily, df_hourly, hourly_box_html


# ===================== 6. GRADIO UI =====================

with gr.Blocks(css=CUSTOM_CSS, title="Hanoi Temperature Forecast") as demo:

    with gr.Column(elem_classes="app-container"):
        gr.Markdown("## Hanoi Temperature Forecast", elem_classes="main-title")
        gr.Markdown(
            "Glass-style dashboard for short-term **daily & hourly** temperature forecast in Hanoi.",
            elem_classes="subtext",
        )

        with gr.Row():
            # LEFT SIDE: 2 GLASS BOXES (STACKED)
            with gr.Column(scale=1):
                # Box 1: Location + last observed
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("#### Hanoi, Viet Nam")

                    last_date, last_temp = latest_observed_temp_daily()
                    gr.Markdown(
                        f"<div class='label-small'>Last observed (daily)</div>"
                        f"<div class='metric-value'>{last_temp:.1f}°C</div>"
                        f"<div class='subtext'>on {last_date}</div>",
                    )
                    gr.Markdown("#### Model Info")
                    gr.Markdown(
                        "Model predicts **daily mean temperature** for multiple days, "
                        "and **next-hour temperature** based on the latest hourly record. ",
                        elem_classes="subtext",
                    )

                # Box 2: mô tả model + Next-hour forecast
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("#### Next Hour Forecast")
                    hourly_box_html = gr.HTML()

            # RIGHT MAIN CARD (controls + summary)
            with gr.Column(scale=2, elem_classes="glass-card"):
                with gr.Row():
                    target_date = gr.Textbox(
                        value=datetime.date.today().isoformat(),
                        label="Start date for daily forecast (YYYY-MM-DD)",
                    )

                    horizon = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Daily horizon (days)",
                    )

                    model_name = gr.Dropdown(
                        choices=["xgboost_v1", "rf_baseline", "lstm_v1"],
                        value="xgboost_v1",
                        label="Model",
                    )

                forecast_btn = gr.Button("Run forecast")
                summary_box = gr.Markdown()
        # DAILY PLOT CARD
        with gr.Row():
            with gr.Column(elem_classes="glass-card-plot"):
                gr.Markdown("#### Daily: actual vs forecast")
                daily_plot = gr.Plot(show_label=False)

        # BOTTOM: DAILY TABLE + HOURLY CARD
        with gr.Row():
            # DAILY TABLE
            with gr.Column(scale=1, elem_classes=["glass-table-card"]):
                gr.Markdown("#### Daily details")
                daily_table = gr.Dataframe(
                    headers=["Date", "Temp (°C)", "Type"],
                    row_count=(0, "dynamic"),
                    col_count=(3, "fixed"),
                    interactive=False,
                    label="",          # ẩn label mặc định
                )

            # HOURLY TABLE
            with gr.Column(scale=1, elem_classes=["glass-table-card"]):
                gr.Markdown("#### Hourly now vs next hour")
                hourly_table = gr.Dataframe(
                    headers=["Datetime", "Temp (°C)", "Type"],
                    row_count=(0, "dynamic"),
                    col_count=(3, "fixed"),
                    interactive=False,
                    label="",
                )

        forecast_btn.click(
            fn=run_forecast,
            inputs=[target_date, horizon, model_name],
            outputs=[summary_box, daily_table, daily_plot, hourly_table, hourly_box_html],
        )

if __name__ == "__main__":
    demo.launch()
