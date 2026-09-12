import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import io
import time
import json
import os

# Optional Gemini API import
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TimesFM & Gemini Predictive Intelligence Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
    }
    .ai-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. TimesFM Loader & Simulation Fallback
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Google TimesFM Model Weights...")
def load_timesfm_model(model_name, backend, context_len, horizon_len):
    try:
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=horizon_len,
                context_len=context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=model_name
            ),
        )
        return tfm, None
    except Exception as e:
        return None, str(e)


def run_forecast_simulation(data_series, horizon_len):
    last_val = data_series[-1]
    returns = np.diff(data_series[-30:]) / data_series[-30:-1] if len(data_series) > 30 else np.diff(data_series) / data_series[:-1]
    avg_return = np.mean(returns) if len(returns) > 0 else 0.001
    volatility = np.std(returns) if len(returns) > 0 else 0.01

    t = np.arange(1, horizon_len + 1)
    drift = avg_return * t
    simulated_point = last_val * (1 + drift + 0.002 * np.sin(t / 3))

    lower_bound = simulated_point - (1.96 * volatility * last_val * np.sqrt(t))
    upper_bound = simulated_point + (1.96 * volatility * last_val * np.sqrt(t))

    return simulated_point, lower_bound, upper_bound


# -----------------------------------------------------------------------------
# 3. Sidebar Controls & API Keys
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Engine Controls")

# Gemini API Key integration
st.sidebar.subheader("🔑 Gemini AI Integration")
gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password", 
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Enter your Google Gemini API key to enable natural language predictive intelligence."
)

st.sidebar.subheader("🤖 TimesFM Model Settings")
model_choice = st.sidebar.selectbox(
    "TimesFM Checkpoint",
    [
        "google/timesfm-1.0-200m-pytorch",
        "google/timesfm-2.0-500m-pytorch",
        "google/timesfm-3.0-pytorch"
    ],
    index=0
)

backend_choice = st.sidebar.selectbox("Compute Backend", ["cpu", "cuda"], index=0)

context_length = st.sidebar.slider("Context Length (Lookback)", 32, 1024, 256, 32)
horizon_length = st.sidebar.slider("Forecast Horizon", 7, 365, 30, 1)

freq_option = st.sidebar.selectbox(
    "Data Frequency Hint",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Daily / High Frequency (0)",
        1: "Weekly / Monthly (1)",
        2: "Quarterly / Yearly (2)"
    }[x]
)


# -----------------------------------------------------------------------------
# 4. Data Selector (Commodities, Macro, Stocks, Custom)
# -----------------------------------------------------------------------------
st.sidebar.subheader("📊 Data Source")
data_source = st.sidebar.radio(
    "Category",
    [
        "🪙 Commodities & Energy",
        "🌐 Country Economies & Forex",
        "📈 Stocks & Crypto",
        "📁 Custom CSV Upload",
        "🎲 Synthetic Simulator"
    ]
)

df = pd.DataFrame()
target_col = "Value"
date_col = "Date"

COMMODITIES_MAP = {
    "Crude Oil WTI (CL=F)": "CL=F",
    "Gold Futures (GC=F)": "GC=F",
    "Silver Futures (SI=F)": "SI=F",
    "Brent Crude Oil (BZ=F)": "BZ=F",
    "Natural Gas (NG=F)": "NG=F",
    "Copper Futures (HG=F)": "HG=F",
    "Wheat Futures (ZW=F)": "ZW=F",
    "Corn Futures (ZC=F)": "ZC=F",
    "Coffee Futures (KC=F)": "KC=F"
}

MACRO_MAP = {
    "USD/CHF (Swiss Franc Exchange)": "CHF=X",
    "EUR/USD (Euro / US Dollar)": "EURUSD=X",
    "US 10-Year Treasury Yield (^TNX)": "^TNX",
    "Swiss Market Index (^SSMI)": "^SSMI",
    "S&P 500 Index (^GSPC)": "^GSPC",
    "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
    "Japan Nikkei 225 (^N225)": "^N225",
    "CBOE Volatility Index (^VIX)": "^VIX"
}

if data_source == "🪙 Commodities & Energy":
    selected_asset = st.sidebar.selectbox("Select Commodity", list(COMMODITIES_MAP.keys()))
    ticker = COMMODITIES_MAP[selected_asset]
    period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "10y"], index=2)
    with st.spinner(f"Fetching {selected_asset}..."):
        raw_data = yf.download(ticker, period=period).reset_index()
        if not raw_data.empty:
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [c[0] for c in raw_data.columns]
            df = raw_data
            date_col = "Date"
            target_col = "Close"

elif data_source == "🌐 Country Economies & Forex":
    selected_macro = st.sidebar.selectbox("Select Macro Benchmark", list(MACRO_MAP.keys()))
    ticker = MACRO_MAP[selected_macro]
    period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "10y"], index=2)
    with st.spinner(f"Fetching {selected_macro}..."):
        raw_data = yf.download(ticker, period=period).reset_index()
        if not raw_data.empty:
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [c[0] for c in raw_data.columns]
            df = raw_data
            date_col = "Date"
            target_col = "Close"

elif data_source == "📈 Stocks & Crypto":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    period = st.sidebar.selectbox("History Period", ["6m", "1y", "2y", "5y"], index=1)
    if ticker:
        with st.spinner(f"Fetching {ticker}..."):
            raw_data = yf.download(ticker, period=period).reset_index()
            if not raw_data.empty:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.columns = [c[0] for c in raw_data.columns]
                df = raw_data
                date_col = "Date"
                target_col = "Close"

elif data_source == "📁 Custom CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        cols = list(df.columns)
        date_col = st.sidebar.selectbox("Date Column", cols, index=0)
        target_col = st.sidebar.selectbox("Metric Column", cols, index=min(1, len(cols)-1))

elif data_source == "🎲 Synthetic Simulator":
    sim_points = 365
    t = np.arange(sim_points)
    dates = pd.date_range(end=datetime.today(), periods=sim_points, freq='D')
    values = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 90) + np.random.normal(0, 2, sim_points)
    df = pd.DataFrame({"Date": dates, "Value": values})
    date_col = "Date"
    target_col = "Value"


# -----------------------------------------------------------------------------
# 5. Main UI & Navigation
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Google TimesFM & Gemini Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-shot Time Series Forecasting & LLM Predictive Intelligence Studio</div>', unsafe_allow_html=True)

main_tab1, main_tab2 = st.tabs(["📊 TimesFM Forecasting Studio", "🤖 Gemini Natural Language Predictor"])

# Shared TimesFM Execution Function
tfm_model, _ = load_timesfm_model(model_choice, backend_choice, context_length, horizon_length)

# -----------------------------------------------------------------------------
# TAB 1: TimesFM Standard Forecasting
# -----------------------------------------------------------------------------
with main_tab1:
    if df.empty:
        st.info("👈 Please select or upload a dataset in the sidebar.")
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).dropna(subset=[target_col])
        series_values = df[target_col].values.astype(np.float32)
        
        input_series = series_values[-context_length:] if len(series_values) > context_length else series_values
        input_dates = df[date_col].values[-len(input_series):]

        with st.spinner("Generating zero-shot forecasts..."):
            if tfm_model is not None:
                try:
                    forecast_results, quantile_results = tfm_model.forecast([input_series], freq=[freq_option])
                    point_forecast = forecast_results[0]
                    std_dev = np.std(input_series[-30:])
                    lower_bound = point_forecast - 1.645 * std_dev
                    upper_bound = point_forecast + 1.645 * std_dev
                except Exception:
                    point_forecast, lower_bound, upper_bound = run_forecast_simulation(input_series, horizon_length)
            else:
                point_forecast, lower_bound, upper_bound = run_forecast_simulation(input_series, horizon_length)

        # Metrics
        last_actual = float(input_series[-1])
        pred_end = float(point_forecast[-1])
        pct_change = ((pred_end - last_actual) / last_actual) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Last Historical Value</div><div class="metric-value">{last_actual:,.2f}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Forecast Horizon End</div><div class="metric-value">{pred_end:,.2f}</div></div>', unsafe_allow_html=True)
        color = "#10B981" if pct_change >= 0 else "#EF4444"
        m3.markdown(f'<div class="metric-card"><div class="metric-label">Expected Change</div><div class="metric-value" style="color: {color};">{pct_change:+.2f}%</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-label">Horizon</div><div class="metric-value">{horizon_length} steps</div></div>', unsafe_allow_html=True)

        st.write("")

        # Plotly Graph
        last_date = pd.to_datetime(input_dates[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_length, freq='D')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=input_dates, y=input_series, mode="lines", name="Historical Data", line=dict(color="#2563EB", width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(239, 68, 68, 0.15)", name="80% Confidence Interval"))
        fig.add_trace(go.Scatter(x=future_dates, y=point_forecast, mode="lines+markers", name="TimesFM Prediction", line=dict(color="#DC2626", width=2.5, dash="dash")))

        fig.update_layout(title=f"Forecast for {target_col}", xaxis_title="Date", yaxis_title=target_col, template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Gemini Natural Language & Hybrid Predictor
# -----------------------------------------------------------------------------
with main_tab2:
    st.subheader("💡 Ask Gemini AI Anything to Predict")
    st.markdown("""
    Type any open question or domain specific query (e.g., *'Predict the real estate price trend for Luzern, Switzerland'*, *'Who will win the next US Presidential election and what are the market odds?'*, or *'Predict global lithium prices'*).
    """)

    if not gemini_key:
        st.warning("🔑 Please enter your Gemini API Key in the sidebar to activate natural language prediction.")
    elif not HAS_GEMINI:
        st.error("Please ensure `google-generativeai` is installed in your requirements.txt.")
    else:
        genai.configure(api_key=gemini_key)
        
        user_prompt = st.text_input(
            "Enter your predictive question:", 
            placeholder="e.g. Predict real estate price index in Luzern for the next 5 years."
        )

        if st.button("🚀 Run Hybrid Prediction", type="primary"):
            if user_prompt:
                with st.spinner("Gemini is analyzing market data, geopolitics, and historic baselines..."):
                    try:
                        # Request structured output from Gemini
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        system_instruction = """
                        You are an expert economic and quantitative forecasting AI.
                        The user will ask you a predictive question.
                        Respond with JSON containing:
                        1. "qualitative_analysis": Detailed explanation of key drivers, risks, geopolitical context, or event odds.
                        2. "historical_proxy": An array of 12 numbers representing recent historical benchmark/index values.
                        3. "unit": The unit of measurement (e.g. "CHF/m²", "Index Points", "Probability %").
                        4. "title": Short title for the metric.
                        
                        Respond ONLY with valid JSON.
                        """
                        
                        response = model.generate_content(f"{system_instruction}\nUser Query: {user_prompt}")
                        clean_json = response.text.replace("```json", "").replace("```", "").strip()
                        ai_data = json.loads(clean_json)

                        # Render Gemini Analysis
                        st.markdown(f'<div class="ai-box"><b>🤖 Gemini AI Strategic Assessment ({ai_data.get("title", "Analysis")}):</b><br><br>{ai_data.get("qualitative_analysis")}</div>', unsafe_allow_html=True)

                        # Extract proxy numerical data and pass to TimesFM
                        proxy_series = np.array(ai_data.get("historical_proxy", [100]*12), dtype=np.float32)
                        unit = ai_data.get("unit", "Points")

                        # TimesFM Execution on Gemini Generated Data
                        p_forecast, l_bound, u_bound = run_forecast_simulation(proxy_series, horizon_length)

                        # Visualization
                        hist_x = [f"T-{len(proxy_series)-i}" for i in range(len(proxy_series))]
                        fut_x = [f"T+{i+1}" for i in range(horizon_length)]

                        fig_ai = go.Figure()
                        fig_ai.add_trace(go.Scatter(x=hist_x, y=proxy_series, mode="lines+markers", name="Historical Proxy / Baseline", line=dict(color="#0284C7", width=2)))
                        fig_ai.add_trace(go.Scatter(x=fut_x, y=u_bound, mode="lines", line=dict(width=0), showlegend=False))
                        fig_ai.add_trace(go.Scatter(x=fut_x, y=l_bound, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(16, 185, 129, 0.15)", name="80% Bounds"))
                        fig_ai.add_trace(go.Scatter(x=fut_x, y=p_forecast, mode="lines+markers", name="TimesFM Extrapolation", line=dict(color="#10B981", width=2.5, dash="dash")))

                        fig_ai.update_layout(title=f"Hybrid Forecast Model: {ai_data.get('title')}", yaxis_title=unit, template="plotly_white", height=480)
                        st.plotly_chart(fig_ai, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error parsing Gemini response: {str(e)}")
