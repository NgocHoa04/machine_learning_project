# style.py

def make_css(bg_url: str) -> str:
    return f"""
/* ========== GLOBAL BACKGROUND ========== */

html, body, #root, .gradio-container {{
    margin: 0;
    background-image: url('{bg_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.gradio-container {{
    background-color: transparent !important;
}}

.app-container {{
    max-width: 1200px;
    margin: 40px auto;
}}

/* ========== GLASS CARDS ========== */

.glass-card {{
    background: rgba(15, 23, 42, 0.55);
    border-radius: 24px;
    padding: 22px 26px;
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    color: #E2E8F0;
}}

.glass-card-plot {{
    background: rgba(15, 23, 42, 0.30);
    border-radius: 24px;
    padding: 18px 24px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    color: #E2E8F0;
}}

/* card riêng cho table */
.glass-table-card {{
    background: rgba(15, 23, 42, 0.55);
    border-radius: 24px;
    padding: 18px 22px;
    border: 1px solid rgba(148, 163, 184, 0.24);
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    color: #E5E7EB;
}}

/* ========== TEXT & TITLES ========== */

.main-title {{
    color: #E2E8F0;
    text-shadow: 0 6px 18px rgba(15, 23, 42, 0.9);
}}

.subtext {{
    color: #CBD5F5;
    font-size: 0.9rem;
}}

.label-small {{
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9CA3AF;
}}

.metric-value {{
    font-size: 1.8rem;
    font-weight: 600;
}}

/* ========== PLOTLY ========== */

.js-plotly-plot,
.js-plotly-plot .plotly,
.js-plotly-plot .plot-container,
.js-plotly-plot .main-svg {{
    background: transparent !important;
}}

/* ========== NEXT-HOUR TEMP BOX ========== */

.next-hour-temp-box {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 10px 16px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(15, 23, 42, 0.20),
        rgba(15, 23, 42, 0.05)
    );
    border: 1px solid rgba(148, 163, 184, 0.40);
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    min-width: 160px;
}}

.next-hour-temp-value {{
    color: #F9FAFB;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

/* ========== TABLE STYLE ĐƠN GIẢN, PHẲNG ========== */

.glass-table-card table {{
    width: 100%;
    border-collapse: collapse;
    background: transparent;
}}

.glass-table-card thead tr {{
    background: rgba(15, 23, 42, 0.95);
}}

.glass-table-card thead tr th {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 1000;
    padding: 6px 10px;
    color: #9CA3AF;
    border-bottom: 1px solid rgba(148, 163, 184, 0.7);
}}

.glass-table-card tbody tr {{
    background: transparent;
}}

.glass-table-card tbody tr:hover {{
    background: rgba(30, 64, 175, 0.35);
}}

.glass-table-card td,
.glass-table-card th {{
    border: none;
    padding: 8px 10px;
}}

.glass-table-card tbody tr td {{
    font-size: 0.85rem;
    color: #E5E7EB;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
}}

.glass-table-card tbody tr th {{
    font-size: 0.85rem;
    font-weight: 1000;
    color: #E5E7EB;
    background: transparent;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    padding: 8px 10px;
}}

.glass-table-card tbody tr:last-child td {{
    border-bottom: none;
}}

.glass-table-card tbody tr td:nth-child(3) {{
    text-align: left;
    font-weight: 500;
}}

.glass-table-card * {{
    border-radius: 0 !important;
    clip-path: none !important;
}}

/* ========== FORM ELEMENTS ========== */

.gradio-container .gradio-input label,
.gradio-container .gradio-slider label {{
    color: #E5E7EB;
}}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {{
    background: rgba(15, 23, 42, 0.75) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.5) !important;
    color: #E5E7EB !important;
}}

.gradio-container button {{
    border-radius: 999px !important;
}}

/* ========== DATETIME / CALENDAR ========== */

/* đảm bảo popup lịch luôn nổi trên cùng và nằm giữa màn hình */
.flatpickr-calendar {{
    position: fixed !important;    
    top: 120px !important;      
    left: 50% !important;
    transform: translateX(-50%) !important;

    z-index: 9999 !important;
    font-family: inherit;
    background: #0f172a !important;
    color: #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
}}

.flatpickr-calendar .flatpickr-day {{
    color: #e5e7eb;
}}

.flatpickr-calendar .flatpickr-day.today {{
    border-color: #60a5fa;
}}

.flatpickr-calendar .flatpickr-day.selected,
.flatpickr-calendar .flatpickr-day.startRange,
.flatpickr-calendar .flatpickr-day.endRange {{
    background: #3b82f6;
    border-color: #3b82f6;
    color: #f9fafb;
}}
"""
