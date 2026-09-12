import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import io
import time

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Google TimesFM Forecasting Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI/UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
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
    .status-box {
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. TimesFM Model Loader (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Google TimesFM Model Weights...")
def load_timesfm_model(model_name, backend, context_len, horizon_len):
    """
    Attempts to load the TimesFM model package.
    Falls back gracefully if the package or HF hub isn't directly reachable.
    """
    try:
        import timesfm
        
        # Initialize TimesFM instance
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
    """
    Fallback simulation engine when TimesFM library is not locally compiled.
    Uses mean-reverting trend + momentum + noise to mock baseline model outputs.
    """
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
# 3. Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Controls")

# --- Model Parameters ---
st.sidebar.subheader("1. TimesFM Model Settings")
model_choice = st.sidebar.selectbox(
    "TimesFM Checkpoint",
    [
        "google/timesfm-1.0-200m-pytorch",
        "google/timesfm-2.0-500m-pytorch",
        "google/timesfm-3.0-pytorch"
    ],
    index=0,
    help="Select pretrained Google Research checkpoint."
)

backend_choice = st.sidebar.selectbox(
    "Compute Backend",
    ["cpu", "cuda"],
    index=0,
    help="Use CUDA for GPU acceleration if PyTorch with CUDA is available."
)

context_length = st.sidebar.slider(
    "Context Length (Lookback)",
    min_value=32,
    max_value=1024,
    value=256,
    step=32,
    help="Number of historical time points given to the model."
)

horizon_length = st.sidebar.slider(
    "Forecast Horizon",
    min_value=7,
    max_value=180,
    value=30,
    step=1,
    help="Number of future steps to predict."
)

freq_option = st.sidebar.selectbox(
    "Data Frequency Hint",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "High Frequency / Daily / Intraday (0)",
        1: "Medium Frequency / Weekly / Monthly (1)",
        2: "Low Frequency / Quarterly / Yearly (2)"
    }[x],
    help="TimesFM input frequency parameter."
)

# --- Data Source Selector ---
st.sidebar.subheader("2. Select Data Source")
data_source = st.sidebar.radio(
    "Category",
    [
        "📈 Stock Market / Crypto",
        "🌐 Economy & Macro",
        "📁 Custom CSV / Excel Upload",
        "🎲 Synthetic Simulator"
    ]
)

# Data fetching containers
df = pd.DataFrame()
target_col = "Value"
date_col = "Date"

if data_source == "📈 Stock Market / Crypto":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="e.g. AAPL, NVDA, TSLA, BTC-USD, ^GSPC")
    period = st.sidebar.selectbox("Historical Period", ["6m", "1y", "2y", "5y"], index=1)
    
    if ticker:
        with st.spinner(f"Fetching {ticker} data from Yahoo Finance..."):
            stock_data = yf.download(ticker, period=period)
            if not stock_data.empty:
                stock_data = stock_data.reset_index()
                # Handle MultiIndex columns if returned by yfinance
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = [col[0] for col in stock_data.columns]
                
                df = stock_data
                date_col = "Date"
                target_col = st.sidebar.selectbox("Target Price Field", ["Close", "Open", "High", "Low", "Volume"])
            else:
                st.sidebar.error("Could not fetch data for ticker.")

elif data_source == "🌐 Economy & Macro":
    macro_asset = st.sidebar.selectbox(
        "Select Macro Benchmark",
        [
            "^TNX (US 10-Yr Treasury Yield)",
            "CL=F (Crude Oil Futures)",
            "GC=F (Gold Futures)",
            "^VIX (CBOE Volatility Index)",
            "^GSPC (S&P 500 Index)"
        ]
    )
    symbol = macro_asset.split(" ")[0]
    with st.spinner(f"Fetching {symbol}..."):
        macro_data = yf.download(symbol, period="2y")
        if not macro_data.empty:
            macro_data = macro_data.reset_index()
            if isinstance(macro_data.columns, pd.MultiIndex):
                macro_data.columns = [col[0] for col in macro_data.columns]
            df = macro_data
            date_col = "Date"
            target_col = "Close"

elif data_source == "📁 Custom CSV / Excel Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Time Series File", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.sidebar.success("File uploaded successfully!")
        cols = list(df.columns)
        date_col = st.sidebar.selectbox("Date Column", cols, index=0)
        target_col = st.sidebar.selectbox("Target Metric Column", cols, index=min(1, len(cols)-1))

elif data_source == "🎲 Synthetic Simulator":
    st.sidebar.subheader("Simulator Settings")
    sim_type = st.sidebar.selectbox("Pattern", ["Trend + Seasonality", "Random Walk", "Sine Wave with Noise"])
    sim_points = st.sidebar.number_input("Historical Length", min_value=100, max_value=2000, value=365)
    
    t = np.arange(sim_points)
    dates = pd.date_range(end=datetime.today(), periods=sim_points, freq='D')
    
    if sim_type == "Trend + Seasonality":
        values = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 365.25) + np.random.normal(0, 2, sim_points)
    elif sim_type == "Random Walk":
        values = 100 + np.cumsum(np.random.normal(0.1, 1.5, sim_points))
    else:
        values = 50 + 15 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1, sim_points)
        
    df = pd.DataFrame({"Date": dates, "Value": values})
    date_col = "Date"
    target_col = "Value"


# -----------------------------------------------------------------------------
# 4. Main Dashboard Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Google TimesFM Prediction Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-shot time-series forecasting foundation model by Google Research</div>', unsafe_allow_html=True)


if df.empty:
    st.info("👈 Please select or upload a dataset from the sidebar to start forecasting.")
    st.stop()

# Data Preprocessing
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col).dropna(subset=[target_col])
series_values = df[target_col].values.astype(np.float32)

# Ensure enough context data
if len(series_values) > context_length:
    input_series = series_values[-context_length:]
    input_dates = df[date_col].values[-context_length:]
else:
    input_series = series_values
    input_dates = df[date_col].values


# -----------------------------------------------------------------------------
# 5. Model Execution & Forecasting
# -----------------------------------------------------------------------------
tfm_model, err = load_timesfm_model(model_choice, backend_choice, context_length, horizon_length)

use_fallback = False
if tfm_model is None:
    use_fallback = True

with st.spinner("Generating zero-shot forecasts..."):
    start_time = time.time()
    
    if not use_fallback:
        try:
            # Forecast using TimesFM model
            forecast_results, quantile_results = tfm_model.forecast(
                [input_series],
                freq=[freq_option]
            )
            point_forecast = forecast_results[0]
            
            # Extract quantiles if available (10th and 90th percentile)
            if quantile_results is not None and len(quantile_results) > 0:
                q = quantile_results[0]
                lower_bound = q[:, 0]
                upper_bound = q[:, -1]
            else:
                std_dev = np.std(input_series[-30:]) if len(input_series) >= 30 else np.std(input_series)
                lower_bound = point_forecast - 1.645 * std_dev
                upper_bound = point_forecast + 1.645 * std_dev
        except Exception as exec_err:
            st.warning(f"TimesFM Execution notice: {exec_err}. Running in baseline projection mode.")
            point_forecast, lower_bound, upper_bound = run_forecast_simulation(input_series, horizon_length)
    else:
        point_forecast, lower_bound, upper_bound = run_forecast_simulation(input_series, horizon_length)
        
    execution_time = round(time.time() - start_time, 3)

# Build Future Dates
last_date = pd.to_datetime(input_dates[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_length, freq='D')

# Status banner
if use_fallback:
    st.warning("⚠️ **Running in Fallback Mode**: The native `timesfm` package is not compiled in this environment or GPU memory is unavailable. Showing baseline prediction projection.")
else:
    st.success(f"✅ **TimesFM Execution Complete**: Generated {horizon_length}-step zero-shot forecast in {execution_time}s.")


# -----------------------------------------------------------------------------
# 6. Analytics & Key Metrics
# -----------------------------------------------------------------------------
last_actual = float(input_series[-1])
pred_end = float(point_forecast[-1])
abs_change = pred_end - last_actual
pct_change = (abs_change / last_actual) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Historical Value</div>
        <div class="metric-value">{last_actual:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecast Horizon End</div>
        <div class="metric-value">{pred_end:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    color = "#10B981" if pct_change >= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expected Growth / Delta</div>
        <div class="metric-value" style="color: {color};">{pct_change:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Prediction Horizon</div>
        <div class="metric-value">{horizon_length} steps</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# 7. Interactive Visualization Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Interactive Forecast Chart", "📋 Forecast Data Table", "⚙️ How to Setup Local TimesFM"])

with tab1:
    # Create Plotly Chart
    fig = go.Figure()

    # Historical Context Line
    fig.add_trace(go.Scatter(
        x=input_dates,
        y=input_series,
        mode="lines",
        name="Historical Data",
        line=dict(color="#2563EB", width=2)
    ))

    # Forecast Upper Bound
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Forecast Lower Bound (Fill Area)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(239, 68, 68, 0.15)",
        name="80% Confidence Band"
    ))

    # Forecast Point Prediction Line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=point_forecast,
        mode="lines+markers",
        name="TimesFM Forecast",
        line=dict(color="#DC2626", width=2.5, dash="dash"),
        marker=dict(size=4)
    ))

    # Vertical Separator
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_width=1,
        line_dash="dot",
        line_color="#64748B",
        annotation_text="Forecast Start",
        annotation_position="top left"
    )

    fig.update_layout(
        title=f"Time Series Forecast ({target_col})",
        xaxis_title="Date",
        yaxis_title=target_col,
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Forecast Data Breakdown")
    
    # Build export dataframe
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Point": point_forecast,
        "Lower_Quantile": lower_bound,
        "Upper_Quantile": upper_bound
    })
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.dataframe(forecast_df.style.format({
            "Forecast_Point": "{:.4f}",
            "Lower_Quantile": "{:.4f}",
            "Upper_Quantile": "{:.4f}"
        }), height=400)
    
    with col_b:
        st.write("### Export Data")
        csv_buffer = io.StringIO()
        forecast_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"timesfm_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with tab3:
    st.markdown("""
    ### Quick Setup Guide for Local TimesFM Environment

    To run Google TimesFM natively on your GPU or CPU machine:

    #### 1. Install Dependencies
    ```bash
    pip install torch torchvision torchaudio
    pip install git+[https://github.com/google-research/timesfm.git](https://github.com/google-research/timesfm.git)
    ```

    #### 2. Run Streamlit App
    ```bash
    streamlit run app.py
    ```

    #### 3. HuggingFace Model Weights Access
    TimesFM automatically downloads public checkpoints from Hugging Face:
    * `google/timesfm-1.0-200m-pytorch`
    * `google/timesfm-2.0-500m-pytorch`
    * `google/timesfm-3.0-pytorch`
    """)
