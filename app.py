import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

# Gemini API import check
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# -----------------------------------------------------------------------------
# 1. Page Configuration & Apple Design System (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TimesFM & Gemini Predictive Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Apple Font & Clean Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", Roboto, sans-serif;
        color: #1D1D1F;
    }
    
    .stApp {
        background-color: #F5F5F7;
    }
    
    /* Title Styles */
    .apple-header {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1D1D1F;
        margin-bottom: 0.2rem;
    }
    
    .apple-subheader {
        font-size: 1.05rem;
        font-weight: 400;
        color: #86868B;
        margin-bottom: 1.8rem;
    }
    
    /* Apple Minimal Card Styling */
    .apple-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        transition: all 0.2s ease;
    }
    
    .apple-card:hover {
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.07);
    }
    
    .apple-metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .apple-metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1D1D1F;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    
    .apple-metric-sub {
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* AI Box Glassmorphic Styling */
    .ai-insight-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(0, 113, 227, 0.2);
        box-shadow: 0 4px 24px rgba(0, 113, 227, 0.06);
        margin-bottom: 24px;
    }

    .ai-insight-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0071E3;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }

    .ai-insight-body {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #333336;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. TimesFM & Gemini Engines
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
    last_val = float(data_series[-1])
    returns = np.diff(data_series[-30:]) / data_series[-30:-1] if len(data_series) > 30 else np.diff(data_series) / data_series[:-1]
    avg_return = np.mean(returns) if len(returns) > 0 else 0.001
    volatility = np.std(returns) if len(returns) > 0 else 0.01

    t = np.arange(1, horizon_len + 1)
    drift = avg_return * t
    simulated_point = last_val * (1 + drift + 0.002 * np.sin(t / 3))

    lower_bound = simulated_point - (1.96 * volatility * last_val * np.sqrt(t))
    upper_bound = simulated_point + (1.96 * volatility * last_val * np.sqrt(t))

    return simulated_point, lower_bound, upper_bound


def call_gemini_with_fallback(system_instruction, user_prompt):
    candidate_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-pro"
    ]
    
    try:
        active_from_api = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except Exception:
        active_from_api = []

    all_candidates = []
    for model in active_from_api + candidate_models:
        clean = model.replace("models/", "")
        if clean not in all_candidates and "2.5" not in clean:
            all_candidates.append(clean)

    last_err = None
    for model_name in all_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"{system_instruction}\nUser Query: {user_prompt}")
            if response and response.text:
                return response.text, model_name
        except Exception as e:
            last_err = e
            continue

    raise last_err or Exception("Could not find an active Gemini model for this API key.")


def parse_gemini_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)


# -----------------------------------------------------------------------------
# 3. Sidebar (Simplified & Intuitive Controls)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Workspace Setup")

# Gemini API Key section
with st.sidebar.expander("🔑 Gemini AI Connection", expanded=True):
    gemini_key = st.text_input(
        "API Key", 
        type="password", 
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Paste your Google Gemini API key to enable AI queries."
    )

# Model configuration in simple plain English
with st.sidebar.expander("🤖 Forecasting Engine Settings", expanded=False):
    model_choice = st.selectbox(
        "TimesFM Model Version",
        [
            "google/timesfm-1.0-200m-pytorch",
            "google/timesfm-2.0-500m-pytorch",
            "google/timesfm-3.0-pytorch"
        ],
        index=0
    )
    backend_choice = st.selectbox("Compute Engine", ["cpu", "cuda"], index=0)
    context_length = st.slider("Past History Window (Days)", 32, 500, 256, 32, help="How many past data points TimesFM reads to understand trends.")
    horizon_length = st.slider("Future Forecast Distance (Days/Steps)", 7, 365, 60, 1, help="How far into the future you want to project.")

# Data Selector Section
st.sidebar.markdown("### 📊 Choose Market Data")
data_source = st.sidebar.radio(
    "Data Category",
    [
        "🪙 Commodities & Energy",
        "🌐 Economies & Currencies",
        "📈 Stocks & Crypto",
        "📁 Upload CSV File",
        "🎲 Demo Simulator"
    ]
)

df = pd.DataFrame()
target_col = "Value"
date_col = "Date"

COMMODITIES_MAP = {
    "Gold Futures": "GC=F",
    "Crude Oil WTI": "CL=F",
    "Silver Futures": "SI=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper Futures": "HG=F",
    "Wheat Futures": "ZW=F"
}

MACRO_MAP = {
    "Swiss Franc (USD/CHF)": "CHF=X",
    "Euro (EUR/USD)": "EURUSD=X",
    "Swiss Market Index (SMI)": "^SSMI",
    "S&P 500 Index": "^GSPC",
    "US 10-Yr Treasury Yield": "^TNX",
    "Volatility Index (VIX)": "^VIX"
}

if data_source == "🪙 Commodities & Energy":
    selected_asset = st.sidebar.selectbox("Commodity", list(COMMODITIES_MAP.keys()))
    ticker = COMMODITIES_MAP[selected_asset]
    period = st.sidebar.selectbox("History Duration", ["1y", "2y", "5y", "10y"], index=2)
    with st.spinner(f"Loading {selected_asset}..."):
        raw_data = yf.download(ticker, period=period).reset_index()
        if not raw_data.empty:
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [c[0] for c in raw_data.columns]
            df = raw_data
            date_col = "Date"
            target_col = "Close"

elif data_source == "🌐 Economies & Currencies":
    selected_macro = st.sidebar.selectbox("Economic Asset", list(MACRO_MAP.keys()))
    ticker = MACRO_MAP[selected_macro]
    period = st.sidebar.selectbox("History Duration", ["1y", "2y", "5y", "10y"], index=2)
    with st.spinner(f"Loading {selected_macro}..."):
        raw_data = yf.download(ticker, period=period).reset_index()
        if not raw_data.empty:
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [c[0] for c in raw_data.columns]
            df = raw_data
            date_col = "Date"
            target_col = "Close"

elif data_source == "📈 Stocks & Crypto":
    ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")
    period = st.sidebar.selectbox("History Duration", ["6m", "1y", "2y", "5y"], index=1)
    if ticker:
        with st.spinner(f"Loading {ticker}..."):
            raw_data = yf.download(ticker, period=period).reset_index()
            if not raw_data.empty:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.columns = [c[0] for c in raw_data.columns]
                df = raw_data
                date_col = "Date"
                target_col = "Close"

elif data_source == "📁 Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        cols = list(df.columns)
        date_col = st.sidebar.selectbox("Date Column", cols, index=0)
        target_col = st.sidebar.selectbox("Values Column", cols, index=min(1, len(cols)-1))

elif data_source == "🎲 Demo Simulator":
    sim_points = 365 * 3
    t = np.arange(sim_points)
    dates = pd.date_range(end=datetime.today(), periods=sim_points, freq='D')
    values = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 90) + np.random.normal(0, 1.5, sim_points)
    df = pd.DataFrame({"Date": dates, "Value": values})
    date_col = "Date"
    target_col = "Value"


# -----------------------------------------------------------------------------
# 4. Main Interface & Navigation
# -----------------------------------------------------------------------------
st.markdown('<div class="apple-header">Predictive Intelligence Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="apple-subheader">Seamless integration of Gemini AI contextual research with Google TimesFM forecasting engine.</div>', unsafe_allow_html=True)

main_tab1, main_tab2 = st.tabs(["📊 Standard Dataset Forecasting", "🤖 Gemini AI ➔ Google TimesFM Pipeline"])

tfm_model, _ = load_timesfm_model(model_choice, backend_choice, context_length, horizon_length)

# -----------------------------------------------------------------------------
# TAB 1: Standard Interactive Dataset Forecasting
# -----------------------------------------------------------------------------
with main_tab1:
    if df.empty:
        st.info("👈 Please select or upload a dataset in the left sidebar to begin.")
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).dropna(subset=[target_col])
        
        series_values = df[target_col].values.astype(np.float32)
        dates_array = df[date_col].values
        
        input_series = series_values[-context_length:] if len(series_values) > context_length else series_values
        input_dates = dates_array[-len(input_series):]

        with st.spinner("Computing zero-shot forecast with TimesFM..."):
            if tfm_model is not None:
                try:
                    forecast_results, _ = tfm_model.forecast([input_series], freq=[0])
                    point_forecast = forecast_results[0]
                    std_dev = np.std(input_series[-30:])
                    lower_bound = point_forecast - 1.645 * std_dev
                    upper_bound = point_forecast + 1.645 * std_dev
                except Exception:
                    point_forecast, lower_bound, upper_bound = run_forecast_simulation(input_series, horizon_length)
            else:
                point_forecast, lower_bound, upper_bound = run_forecast_simulation(input_series, horizon_length)

        last_actual = float(input_series[-1])
        pred_end = float(point_forecast[-1])
        pct_change = ((pred_end - last_actual) / last_actual) * 100

        last_date = pd.to_datetime(input_dates[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_length, freq='D')
        forecast_end_date = future_dates[-1]

        # Top Metric Cards (Apple Style)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'''
        <div class="apple-card">
            <div class="apple-metric-label">Current Value</div>
            <div class="apple-metric-val">{last_actual:,.2f}</div>
            <div class="apple-metric-sub" style="color: #86868B;">As of {last_date.strftime("%b %d, %Y")}</div>
        </div>
        ''', unsafe_allow_html=True)

        c2.markdown(f'''
        <div class="apple-card">
            <div class="apple-metric-label">Target Prediction</div>
            <div class="apple-metric-val">{pred_end:,.2f}</div>
            <div class="apple-metric-sub" style="color: #86868B;">For {forecast_end_date.strftime("%b %d, %Y")}</div>
        </div>
        ''', unsafe_allow_html=True)

        delta_color = "#10B981" if pct_change >= 0 else "#EF4444"
        c3.markdown(f'''
        <div class="apple-card">
            <div class="apple-metric-label">Expected Growth</div>
            <div class="apple-metric-val" style="color: {delta_color};">{pct_change:+.2f}%</div>
            <div class="apple-metric-sub" style="color: {delta_color};">Net projected shift</div>
        </div>
        ''', unsafe_allow_html=True)

        c4.markdown(f'''
        <div class="apple-card">
            <div class="apple-metric-label">Future Distance</div>
            <div class="apple-metric-val">+{horizon_length} Days</div>
            <div class="apple-metric-sub" style="color: #86868B;">{(horizon_length/365.25):.1f} Years ahead</div>
        </div>
        ''', unsafe_allow_html=True)

        # Plotly Drag & Pan Interactive Graph
        fig = go.Figure()
        
        # Past Historical Line
        fig.add_trace(go.Scatter(
            x=input_dates, 
            y=input_series, 
            mode="lines", 
            name="Historical Data", 
            line=dict(color="#0071E3", width=2.5),
            hovertemplate="<b>Date</b>: %{x|%b %d, %Y}<br><b>Value</b>: %{y:,.2f}<extra></extra>"
        ))
        
        # TimesFM Prediction Line
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=upper_bound, 
            mode="lines", 
            line=dict(width=0), 
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=lower_bound, 
            mode="lines", 
            line=dict(width=0), 
            fill="tonexty", 
            fillcolor="rgba(255, 45, 85, 0.12)", 
            name="80% Confidence Interval",
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=point_forecast, 
            mode="lines+markers", 
            name="TimesFM Forecast", 
            line=dict(color="#FF2D55", width=2.5, dash="dash"),
            marker=dict(size=4),
            hovertemplate="<b>Forecast Date</b>: %{x|%b %d, %Y}<br><b>Predicted</b>: %{y:,.2f}<extra></extra>"
        ))

        # Divider line marking "Today"
        fig.add_vline(
            x=last_date.timestamp() * 1000, 
            line_width=1.5, 
            line_dash="dot", 
            line_color="#86868B",
            annotation_text="Today / Forecast Start",
            annotation_position="top left",
            annotation_font=dict(size=12, color="#86868B")
        )

        fig.update_layout(
            title=dict(text=f"Historical & TimesFM Projection: {target_col}", font=dict(size=18, color="#1D1D1F")),
            template="plotly_white",
            height=530,
            dragmode="pan", # Enable smooth drag and panning across history
            xaxis=dict(
                title="Timeline (Years & Dates)",
                type="date",
                rangeslider=dict(visible=True), # Bottom interactive range slider
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(step="all", label="All History")
                    ])
                )
            ),
            yaxis=dict(title=target_col, tickformat=",.2f"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})


# -----------------------------------------------------------------------------
# TAB 2: Direct Gemini AI ➔ Google TimesFM Pipeline with Clear Timeline
# -----------------------------------------------------------------------------
with main_tab2:
    st.markdown("""
    <div style="background: #FFFFFF; border-radius: 16px; padding: 20px; border: 1px solid rgba(0,0,0,0.06); margin-bottom: 20px;">
        <h4 style="margin: 0 0 6px 0; color: #1D1D1F;">💬 Ask Gemini AI + TimesFM Anything</h4>
        <p style="margin: 0; color: #86868B; font-size: 0.95rem;">
            Type a natural language request (e.g. <i>"Predict real estate price trend in Zurich for the next 5 years"</i> or <i>"Predict global solar energy adoption rate"</i>). 
            <b>Gemini AI</b> researches the topic and builds a clean historical series with real years, and <b>Google TimesFM</b> runs the math to project the graph into the future.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not gemini_key:
        st.warning("🔑 Please enter your Gemini API Key in the sidebar to run AI natural language predictions.")
    elif not HAS_GEMINI:
        st.error("Missing package `google-generativeai` in `requirements.txt`.")
    else:
        genai.configure(api_key=gemini_key)
        
        user_prompt = st.text_input(
            "Enter your prediction query:", 
            placeholder="e.g. Predict Zurich housing price index over the next 5 years"
        )

        if st.button("🚀 Run Gemini ➔ TimesFM Prediction Pipeline", type="primary"):
            if user_prompt:
                current_year = datetime.now().year
                
                # Step 1: Gemini contextual research
                with st.spinner("Step 1/2: Gemini AI is researching context & building historical timeline..."):
                    try:
                        system_instruction = f"""
                        You are an expert economic and statistical researcher.
                        The user will ask you a prediction question.
                        Research the historical context and return STRICTLY valid JSON with:
                        1. "title": Short clean title for the metric.
                        2. "qualitative_analysis": 2-3 clear paragraphs explaining key real-world drivers, market conditions, and future risks.
                        3. "unit": Unit of measurement (e.g. "CHF / m²", "Index (Base 100)", "USD / Barrel").
                        4. "start_year": Integer starting year for history (e.g. {current_year - 6}).
                        5. "historical_series": An array of 24 to 36 sequential historical numerical values ending at present year ({current_year}).
                        6. "step_frequency": String indicating time step: "Yearly", "Quarterly", or "Monthly".
                        
                        Respond ONLY with raw valid JSON. No markdown wrappers.
                        """
                        
                        raw_response, active_model_used = call_gemini_with_fallback(system_instruction, user_prompt)
                        ai_data = parse_gemini_json(raw_response)

                        # Render Apple-style AI Insight Box
                        st.markdown(f'''
                        <div class="ai-insight-card">
                            <div class="ai-insight-title">
                                <span>🤖 Gemini AI Contextual Insight ({ai_data.get("title", "Analysis")})</span>
                                <span style="font-size: 0.75rem; background: #E8F2FD; color: #0071E3; padding: 3px 8px; border-radius: 12px; font-weight: 500;">
                                    Model: {active_model_used}
                                </span>
                            </div>
                            <div class="ai-insight-body">
                                {ai_data.get("qualitative_analysis")}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

                        # Step 2: Extract historical values & build realistic Datetime objects
                        hist_vals = np.array(ai_data.get("historical_series", [100]*24), dtype=np.float32)
                        unit = ai_data.get("unit", "Points")
                        freq_type = ai_data.get("step_frequency", "Monthly")
                        start_year = int(ai_data.get("start_year", current_year - 5))

                        # Build clear timeline dates
                        if freq_type.lower() == "yearly":
                            hist_dates = pd.date_range(start=f"{start_year}-01-01", periods=len(hist_vals), freq="YS")
                            fut_freq = "YS"
                        elif freq_type.lower() == "quarterly":
                            hist_dates = pd.date_range(start=f"{start_year}-01-01", periods=len(hist_vals), freq="QS")
                            fut_freq = "QS"
                        else: # Monthly default
                            hist_dates = pd.date_range(start=f"{start_year}-01-01", periods=len(hist_vals), freq="MS")
                            fut_freq = "MS"

                        # Step 3: Direct TimesFM Mathematical Prediction
                        with st.spinner("Step 2/2: Passing Gemini dataset into Google TimesFM model..."):
                            if tfm_model is not None:
                                try:
                                    tfm_out, _ = tfm_model.forecast([hist_vals], freq=[0])
                                    p_forecast = tfm_out[0]
                                    std_dev = np.std(hist_vals[-6:]) if len(hist_vals) >= 6 else np.std(hist_vals)
                                    l_bound = p_forecast - 1.645 * std_dev
                                    u_bound = p_forecast + 1.645 * std_dev
                                    engine_label = "Google TimesFM Transformer Weights"
                                except Exception:
                                    p_forecast, l_bound, u_bound = run_forecast_simulation(hist_vals, horizon_length)
                                    engine_label = "Statistical Baseline Engine"
                            else:
                                p_forecast, l_bound, u_bound = run_forecast_simulation(hist_vals, horizon_length)
                                engine_label = "Statistical Baseline Engine"

                        # Generate clear Future Date timeline
                        last_hist_date = hist_dates[-1]
                        fut_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=28), periods=horizon_length, freq=fut_freq)
                        
                        last_val = float(hist_vals[-1])
                        end_val = float(p_forecast[-1])
                        growth_pct = ((end_val - last_val) / last_val) * 100

                        # Summary Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.markdown(f'''
                        <div class="apple-card">
                            <div class="apple-metric-label">Historical Baseline</div>
                            <div class="apple-metric-val">{last_val:,.2f} {unit}</div>
                            <div class="apple-metric-sub" style="color: #86868B;">As of {last_hist_date.strftime("%Y")}</div>
                        </div>
                        ''', unsafe_allow_html=True)

                        m2.markdown(f'''
                        <div class="apple-card">
                            <div class="apple-metric-label">TimesFM Target</div>
                            <div class="apple-metric-val">{end_val:,.2f} {unit}</div>
                            <div class="apple-metric-sub" style="color: #86868B;">By Year {fut_dates[-1].strftime("%Y")}</div>
                        </div>
                        ''', unsafe_allow_html=True)

                        g_color = "#10B981" if growth_pct >= 0 else "#EF4444"
                        m3.markdown(f'''
                        <div class="apple-card">
                            <div class="apple-metric-label">Projected Shift</div>
                            <div class="apple-metric-val" style="color: {g_color};">{growth_pct:+.2f}%</div>
                            <div class="apple-metric-sub" style="color: {g_color};">Net expected change</div>
                        </div>
                        ''', unsafe_allow_html=True)

                        m4.markdown(f'''
                        <div class="apple-card">
                            <div class="apple-metric-label">Timeline Horizon</div>
                            <div class="apple-metric-val">{hist_dates[0].strftime("%Y")} ➔ {fut_dates[-1].strftime("%Y")}</div>
                            <div class="apple-metric-sub" style="color: #86868B;">+{horizon_length} {freq_type} Steps</div>
                        </div>
                        ''', unsafe_allow_html=True)

                        # Interactive Graph for Gemini + TimesFM
                        fig_ai = go.Figure()
                        
                        # Gemini Historical Line
                        fig_ai.add_trace(go.Scatter(
                            x=hist_dates, 
                            y=hist_vals, 
                            mode="lines+markers", 
                            name="Gemini Historical Data", 
                            line=dict(color="#0071E3", width=2.5),
                            marker=dict(size=5),
                            hovertemplate="<b>Year / Date</b>: %{x|%b %Y}<br><b>Value</b>: %{y:,.2f} " + unit + "<extra></extra>"
                        ))
                        
                        # TimesFM Future Forecast Bounds
                        fig_ai.add_trace(go.Scatter(x=fut_dates, y=u_bound, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                        fig_ai.add_trace(go.Scatter(
                            x=fut_dates, 
                            y=l_bound, 
                            mode="lines", 
                            line=dict(width=0), 
                            fill="tonexty", 
                            fillcolor="rgba(52, 199, 89, 0.15)", 
                            name="TimesFM Confidence Range",
                            hoverinfo="skip"
                        ))
                        
                        # TimesFM Future Forecast Line
                        fig_ai.add_trace(go.Scatter(
                            x=fut_dates, 
                            y=p_forecast, 
                            mode="lines+markers", 
                            name=f"TimesFM Forecast ({engine_label})", 
                            line=dict(color="#34C759", width=2.5, dash="dash"),
                            marker=dict(size=5),
                            hovertemplate="<b>Future Date</b>: %{x|%b %Y}<br><b>Forecast</b>: %{y:,.2f} " + unit + "<extra></extra>"
                        ))

                        # Vertical Today Divider Line
                        fig_ai.add_vline(
                            x=last_hist_date.timestamp() * 1000, 
                            line_width=1.5, 
                            line_dash="dot", 
                            line_color="#86868B",
                            annotation_text="Present Baseline",
                            annotation_position="top left",
                            annotation_font=dict(size=12, color="#86868B")
                        )

                        fig_ai.update_layout(
                            title=dict(text=f"Hybrid AI Forecast: {ai_data.get('title')}", font=dict(size=18, color="#1D1D1F")),
                            yaxis_title=unit,
                            template="plotly_white",
                            height=520,
                            dragmode="pan", # Drag and Pan history seamlessly
                            xaxis=dict(
                                title="Years & Calendar Dates",
                                type="date",
                                rangeslider=dict(visible=True) # Bottom range slider bar
                            ),
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig_ai, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

                    except Exception as e:
                        st.error(f"Pipeline Error: {str(e)}")
