import requests
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# ===================== CONFIG & STYLING =====================
st.set_page_config(
    page_title="📊 Forecast Bitcoin", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk desain modern
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #00ff88 !important;
        text-shadow: 0 0 20px rgba(0,255,136,0.5);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00ff88 !important;
        text-shadow: 0 0 10px rgba(0,255,136,0.3);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 15px rgba(240,147,251,0.3);
    }
    
    .section-header h2 {
        color: white !important;
        margin: 0 !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input styling */
    .stNumberInput input, .stSelectbox select, .stSlider {
        border-radius: 10px;
        border: 2px solid rgba(102,126,234,0.3);
        background: rgba(255,255,255,0.05);
        color: white;
    }
    
    /* Accuracy metrics container */
    .accuracy-container {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .accuracy-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Animation */
    @keyframes glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .glow-effect {
        animation: glow 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Header dengan desain modern
st.markdown("""
<div class="main-header">
    <h1>🚀 Bitcoin Forecast Analytics</h1>
    <p>Real-time BTC/USD Price Prediction with ARIMA & SARIMA Models</p>
</div>
""", unsafe_allow_html=True)

# Auto refresh tiap 30 detik
st_autorefresh(interval=30 * 1000, key="btc_refresh")

# ===================== Sidebar =====================
st.sidebar.markdown("### ⚙️ Model Configuration")
st.sidebar.markdown("---")

col_p, col_d = st.sidebar.columns(2)
with col_p:
    p = st.number_input("📊 p (AR)", min_value=0, max_value=10, value=5, step=1)
with col_d:
    d = st.number_input("📈 d (Diff)", min_value=0, max_value=5, value=1, step=1)

q = st.sidebar.number_input("📉 q (MA)", min_value=0, max_value=10, value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Forecast Settings")

steps_forecast = st.sidebar.slider(
    "📈 Prediction Steps", 
    min_value=5, max_value=200, value=30, step=5,
    help="Number of future steps to predict"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Time Interval")

# Pilihan interval
intervals = ["15 Menit", "30 Menit", "1 Jam", "1 Hari", "1 Bulan"]
choice = st.selectbox("📅 Select Interval", intervals, index=3)

# Atur days & resample rule sesuai interval
if choice in ["15 Menit", "30 Menit", "1 Jam"]:
    days = "1"
    rule = {
        "15 Menit": "15min",
        "30 Menit": "30min",
        "1 Jam": "1H"
    }[choice]
elif choice == "1 Hari":
    days = "30"
    rule = "4H"
elif choice == "1 Bulan":
    days = "365"
    rule = "1D"

st.sidebar.markdown("---")
st.sidebar.info(f"🔄 Auto-refresh: Every 30 seconds\n\n📍 Timezone: WIB (Asia/Jakarta)")

# Panggil API CoinGecko
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": days}

with st.spinner("🔄 Fetching data from CoinGecko..."):
    try:
        res = requests.get(url, params=params, timeout=30)
        data = res.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        st.stop()

if "prices" in data:
    # Konversi ke DataFrame dan ubah ke timezone WIB
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")

    # Resample ke OHLC
    ohlc = df.resample(rule, on="timestamp").ohlc().dropna()

    # ==================== CANDLESTICK CHART ====================
    st.markdown(f"""
    <div class="section-header">
        <h2>📊 Candlestick Chart - {choice}</h2>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(data=[go.Candlestick(
        x=ohlc.index,
        open=ohlc["price"]["open"],
        high=ohlc["price"]["high"],
        low=ohlc["price"]["low"],
        close=ohlc["price"]["close"],
        name="BTC/USD",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff0066",
        increasing_fillcolor="#00ff88",
        decreasing_fillcolor="#ff0066",
        increasing_line_width=2,
        decreasing_line_width=2
    )])

    chart_width = 2500 if choice != "1 Hari" else 1500
    fig.update_layout(
        title={
            'text': f"BTC/USD - Candlestick ({choice})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis_title="Time (WIB)",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        width=chart_width,
        height=900,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(15,12,41,0.8)',
        paper_bgcolor='rgba(15,12,41,0.5)',
        font=dict(color='white', size=12),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        )
    )

    if choice == "1 Hari" and len(ohlc) > 100:
        fig.update_xaxes(range=[ohlc.index[-100], ohlc.index[-1]])

    fig.update_xaxes(rangeslider_visible=False, fixedrange=False)

    st.plotly_chart(fig, config={'displayModeBar': True, 'responsive': True})

    # ==================== METRICS ====================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Current Price", f"${df['price'].iloc[-1]:,.2f}")
    with col2:
        price_change = df['price'].iloc[-1] - df['price'].iloc[0]
        st.metric("📈 Change", f"${price_change:,.2f}", f"{(price_change/df['price'].iloc[0]*100):.2f}%")
    with col3:
        st.metric("📊 High", f"${df['price'].max():,.2f}")
    with col4:
        st.metric("📉 Low", f"${df['price'].min():,.2f}")

    # ==================== OHLC DATASET ====================
    st.markdown("""
    <div class="section-header">
        <h2>📑 OHLC Dataset</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(ohlc, width='stretch', height=400)

    # ==================== CLOSING PRICE CHART ====================
    st.markdown("""
    <div class="section-header">
        <h2>📈 Closing Price Trend</h2>
    </div>
    """, unsafe_allow_html=True)
    
    plt.style.use('dark_background')
    fig_close, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ohlc.index, ohlc["price"]["close"], label="Closing Price", color="#00ff88", linewidth=2)
    ax.fill_between(ohlc.index, ohlc["price"]["close"], alpha=0.3, color="#00ff88")
    ax.set_title("BTC/USD Closing Price", fontsize=18, fontweight='bold', color="white", pad=20)
    ax.set_xlabel("Time (WIB)", fontsize=14, color="white")
    ax.set_ylabel("Price (USD)", fontsize=14, color="white")
    ax.legend(facecolor="#1a1a2e", edgecolor="#00ff88", fontsize=12)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_facecolor('#0f0c29')
    fig_close.patch.set_facecolor('#0f0c29')
    plt.tight_layout()
    st.pyplot(fig_close)

    # ==================== ARIMA PREDICTION ====================
    st.markdown("""
    <div class="section-header">
        <h2>🔮 ARIMA Prediction Model</h2>
    </div>
    """, unsafe_allow_html=True)

    series = ohlc["price"]["close"]

    # ==================== ADF TEST ====================
    st.markdown("""
    <div class="section-header">
        <h2>📊 ADF Test (Stationarity Check)</h2>
    </div>
    """, unsafe_allow_html=True)

    def perform_adf_test(timeseries, title=''):
        """
        Perform Augmented Dickey-Fuller test
        """
        result = adfuller(timeseries, autolag='AIC')
        
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        return {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': p_value < 0.05
        }

    # Test original series
    adf_original = perform_adf_test(series, 'Original Series')
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin: 1rem 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);">
        <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">
            🔍 ADF Test Results - Original Series
        </h3>
    """, unsafe_allow_html=True)
    
    col_adf1, col_adf2, col_adf3 = st.columns(3)
    
    with col_adf1:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                    border-radius: 10px; backdrop-filter: blur(10px);">
            <h4 style="color: white; margin: 0;">ADF Statistic</h4>
            <p style="font-size: 2rem; color: #00ff88; font-weight: bold; margin: 0.5rem 0;">
                {adf_original['adf_statistic']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_adf2:
        p_color = "#00ff88" if adf_original['p_value'] < 0.05 else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                    border-radius: 10px; backdrop-filter: blur(10px);">
            <h4 style="color: white; margin: 0;">p-value</h4>
            <p style="font-size: 2rem; color: {p_color}; font-weight: bold; margin: 0.5rem 0;">
                {adf_original['p_value']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_adf3:
        status = "✅ STATIONARY" if adf_original['is_stationary'] else "⚠️ NON-STATIONARY"
        status_color = "#00ff88" if adf_original['is_stationary'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                    border-radius: 10px; backdrop-filter: blur(10px);">
            <h4 style="color: white; margin: 0;">Status</h4>
            <p style="font-size: 1.5rem; color: {status_color}; font-weight: bold; margin: 0.5rem 0;">
                {status}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Critical Values
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📌 Critical Values:")
    col_cv1, col_cv2, col_cv3 = st.columns(3)
    
    with col_cv1:
        cv_1_color = "#00ff88" if adf_original['adf_statistic'] < adf_original['critical_values']['1%'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; 
                    border-radius: 8px; text-align: center;">
            <strong style="color: white;">1% Level</strong><br>
            <span style="font-size: 1.3rem; color: {cv_1_color};">
                {adf_original['critical_values']['1%']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_cv2:
        cv_5_color = "#00ff88" if adf_original['adf_statistic'] < adf_original['critical_values']['5%'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; 
                    border-radius: 8px; text-align: center;">
            <strong style="color: white;">5% Level</strong><br>
            <span style="font-size: 1.3rem; color: {cv_5_color};">
                {adf_original['critical_values']['5%']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_cv3:
        cv_10_color = "#00ff88" if adf_original['adf_statistic'] < adf_original['critical_values']['10%'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; 
                    border-radius: 8px; text-align: center;">
            <strong style="color: white;">10% Level</strong><br>
            <span style="font-size: 1.3rem; color: {cv_10_color};">
                {adf_original['critical_values']['10%']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Interpretation
    if adf_original['is_stationary']:
        st.success("""
        ✅ **Data is STATIONARY** - The time series does not have a unit root and is suitable for ARIMA modeling.
        The p-value is less than 0.05, rejecting the null hypothesis of non-stationarity.
        """)
    else:
        st.warning(f"""
        ⚠️ **Data is NON-STATIONARY** - The time series has a unit root and needs differencing.
        
        **Recommendations:**
        - Current differencing parameter (d) = {d}
        - Try increasing the 'd' parameter in the sidebar (d=1 or d=2)
        - The model will automatically apply differencing based on your 'd' value
        """)
    
    # Test after differencing if d > 0
    if d > 0:
        series_diff = series.diff(d).dropna()
        adf_diff = perform_adf_test(series_diff, f'After {d} Differencing')
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 15px; margin: 1rem 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);">
            <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">
                🔍 ADF Test Results - After {d} Differencing
            </h3>
        """, unsafe_allow_html=True)
        
        col_diff1, col_diff2, col_diff3 = st.columns(3)
        
        with col_diff1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4 style="color: white; margin: 0;">ADF Statistic</h4>
                <p style="font-size: 2rem; color: #00ff88; font-weight: bold; margin: 0.5rem 0;">
                    {adf_diff['adf_statistic']:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_diff2:
            p_color_diff = "#00ff88" if adf_diff['p_value'] < 0.05 else "#ff0066"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4 style="color: white; margin: 0;">p-value</h4>
                <p style="font-size: 2rem; color: {p_color_diff}; font-weight: bold; margin: 0.5rem 0;">
                    {adf_diff['p_value']:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_diff3:
            status_diff = "✅ STATIONARY" if adf_diff['is_stationary'] else "⚠️ NON-STATIONARY"
            status_color_diff = "#00ff88" if adf_diff['is_stationary'] else "#ff0066"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4 style="color: white; margin: 0;">Status</h4>
                <p style="font-size: 1.5rem; color: {status_color_diff}; font-weight: bold; margin: 0.5rem 0;">
                    {status_diff}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if adf_diff['is_stationary']:
            st.success(f"""
            ✅ **Data is now STATIONARY after {d} differencing!** - Ready for ARIMA modeling.
            """)
        else:
            st.warning(f"""
            ⚠️ **Data is still NON-STATIONARY after {d} differencing** - Consider increasing 'd' parameter.
            """)

    st.markdown("---")

    # Continue with ARIMA modeling
    st.markdown("""
    <div class="section-header">
        <h2>🤖 ARIMA Model Training & Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("🤖 Training ARIMA model..."):
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
        
            forecast = model_fit.forecast(steps=steps_forecast)
            future_index = pd.date_range(start=series.index[-1], periods=steps_forecast+1, freq=rule)[1:]
            forecast_series = pd.Series(forecast, index=future_index)

        # Plot prediksi ARIMA
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=series.index, y=series,
            mode="lines", name="Actual Price", 
            line=dict(color="#667eea", width=3)
        ))
        fig_pred.add_trace(go.Scatter(
            x=forecast_series.index, y=forecast_series,
            mode="lines+markers", name="ARIMA Forecast", 
            line=dict(color="#f093fb", dash="dash", width=3),
            marker=dict(size=6, color="#f093fb")
        ))

        fig_pred.update_layout(
            title={
                'text': f"ARIMA ({p},{d},{q}) Forecast - {steps_forecast} Steps Ahead",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22, 'color': 'white', 'family': 'Arial Black'}
            },
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            width=1500,
            height=600,
            plot_bgcolor='rgba(15,12,41,0.8)',
            paper_bgcolor='rgba(15,12,41,0.5)',
            hovermode='x unified',
            legend=dict(
                bgcolor='rgba(26,26,46,0.8)',
                bordercolor='#00ff88',
                borderwidth=2
            )
        )

        st.plotly_chart(fig_pred, config={'displayModeBar': True, 'responsive': True})

        def forecast_accuracy(forecast, actual):
            forecast = np.array(forecast)
            actual = np.array(actual)
            mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
            me = np.mean(forecast - actual)
            mae = np.mean(np.abs(forecast - actual))
            mpe = np.mean((forecast - actual) / actual)
            rmse = np.mean((forecast - actual) ** 2) ** 0.5
            corr = np.corrcoef(forecast, actual)[0, 1]
            mins = np.amin(np.column_stack([forecast, actual]), axis=1)
            maxs = np.amax(np.column_stack([forecast, actual]), axis=1)
            minmax = 1 - np.mean(mins / maxs)
            acf1 = acf(forecast - actual)[1]
            return {
                'mape': mape, 'me': me, 'mae': mae,
                'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
                'corr': corr, 'minmax': minmax
            }
        
        # Evaluasi Model
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        model_eval = ARIMA(train, order=(p, d, q)).fit()
        predictions = model_eval.forecast(steps=len(test))
        mse = mean_squared_error(test, predictions)

        # MSE Metric
        col_mse1, col_mse2, col_mse3 = st.columns([1,2,1])
        with col_mse2:
            st.metric("📉 Mean Squared Error (MSE)", f"{mse:,.2f}")

        # Tabel prediksi ARIMA
        st.markdown("#### 📊 Forecast Data")
        st.dataframe(forecast_series, width='stretch')

        # Metrik akurasi ARIMA
        accuracy_metrics = forecast_accuracy(predictions, test.values)

        st.markdown("""
        <div class="accuracy-container">
            <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">📊 ARIMA Accuracy Metrics</h3>
        """, unsafe_allow_html=True)
        
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        
        with col_a1:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>MAPE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['mape']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>MAE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['mae']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a2:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>ME</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['me']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>MPE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['mpe']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a3:
            
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>RMSE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['rmse']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>ACF1</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['acf1']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a4:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>Correlation</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['corr']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>Min-Max Error</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics['minmax']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # ==================== SARIMA PREDICTION ====================
        st.markdown("""
        <div class="section-header">
            <h2>🎯 SARIMA Prediction Model</h2>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            with st.spinner("🤖 Training SARIMA model..."):
                model_sarima = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                model_sarima_fit = model_sarima.fit(disp=False)
                sarima_forecast = model_sarima_fit.forecast(steps=steps_forecast)
                sarima_forecast_series = pd.Series(sarima_forecast, index=future_index)
       
            # Plot prediksi SARIMA
            fig_sarima = go.Figure()
            fig_sarima.add_trace(go.Scatter(
                x=series.index, y=series,
                mode="lines", name="Actual Price", 
                line=dict(color="#667eea", width=3)
            ))
            fig_sarima.add_trace(go.Scatter(
                x=sarima_forecast_series.index, y=sarima_forecast_series,
                mode="lines+markers", name="SARIMA Forecast", 
                line=dict(color="#00ff88", dash="dash", width=3),
                marker=dict(size=6, color="#00ff88")
            ))

            fig_sarima.update_layout(
                title={
                    'text': f"SARIMA ({p},{d},{q})(1,1,1,12) Forecast - {steps_forecast} Steps Ahead",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 22, 'color': 'white', 'family': 'Arial Black'}
                },
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                width=1500,
                height=600,
                plot_bgcolor='rgba(15,12,41,0.8)',
                paper_bgcolor='rgba(15,12,41,0.5)',
                hovermode='x unified',
                legend=dict(
                    bgcolor='rgba(26,26,46,0.8)',
                    bordercolor='#00ff88',
                    borderwidth=2
                )
            )

            st.plotly_chart(fig_sarima, config={'displayModeBar': True, 'responsive': True})

            # Tabel prediksi SARIMA
            st.markdown("#### 📊 Forecast Data (SARIMA)")
            st.dataframe(sarima_forecast_series, width='stretch')

            # Metrik akurasi SARIMA
            accuracy_metrics_sarima = forecast_accuracy(model_sarima_fit.fittedvalues[train_size:], test.values)
            
            st.markdown("""
            <div class="accuracy-container">
                <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">📊 SARIMA Accuracy Metrics</h3>
            """, unsafe_allow_html=True)
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>MAPE</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['mape']:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>MAE</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['mae']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s2:
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>ME</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['me']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>MPE</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['mpe']:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s3:
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>RMSE</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['rmse']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>ACF1</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['acf1']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s4:
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>Correlation</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['corr']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="accuracy-item">
                    <strong>Min-Max Error</strong><br>
                    <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_metrics_sarima['minmax']:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

             # Evaluasi Model SARIMA
            try:
                # Hitung MSE untuk SARIMA
                mse_sarima = mean_squared_error(test, sarima_forecast[:len(test)])
                
                # Tampilkan MSE di Streamlit
                col_mse_s1, col_mse_s2, col_mse_s3 = st.columns([1, 2, 1])
                with col_mse_s2:
                    st.metric("📉 SARIMA Mean Squared Error (MSE)", f"{mse_sarima:,.2f}")
            except Exception as e:
                st.error(f"❌ Gagal menghitung MSE untuk SARIMA: {e}")

        except Exception as e:
            st.error(f"❌ SARIMA Error: {e}")

    except Exception as e:
        st.error(f"❌ ARIMA Error: {e}")

else:
    st.error("❌ CoinGecko API did not return price data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6); padding: 2rem;">
    <p>🚀 Built with Streamlit | 📊 Data from CoinGecko API | 🤖 Powered by ARIMA & SARIMA Models</p>
</div>
""", unsafe_allow_html=True)
