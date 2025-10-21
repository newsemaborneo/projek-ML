import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
from model import (
    fetch_bitcoin_data,
    prepare_ohlc_data,
    perform_adf_test,
    run_arima_model,
    run_sarima_model
)

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

# ===================== FETCH DATA =====================
with st.spinner("🔄 Fetching data from CoinGecko..."):
    data = fetch_bitcoin_data(days)

if data and "prices" in data:
    df, ohlc = prepare_ohlc_data(data, rule)
    
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
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    )

    if choice == "1 Hari" and len(ohlc) > 100:
        fig.update_xaxes(range=[ohlc.index[-100], ohlc.index[-1]])

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
    series = ohlc["price"]["close"]
    
    st.markdown("""
    <div class="section-header">
        <h2>📊 ADF Test (Stationarity Check)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Perform ADF test and display results
    adf_original = perform_adf_test(series, d)
    
    # ==================== ARIMA MODEL ====================
    st.markdown("""
    <div class="section-header">
        <h2>🔮 ARIMA Prediction Model</h2>
    </div>
    """, unsafe_allow_html=True)
    
    arima_results = run_arima_model(series, p, d, q, steps_forecast, rule)
    
    if arima_results:
        # Display ARIMA forecast chart
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(
            x=series.index, y=series,
            mode="lines", name="Actual Price", 
            line=dict(color="#667eea", width=3)
        ))
        fig_arima.add_trace(go.Scatter(
            x=arima_results['forecast_series'].index, 
            y=arima_results['forecast_series'],
            mode="lines+markers", name="ARIMA Forecast", 
            line=dict(color="#f093fb", dash="dash", width=3),
            marker=dict(size=6, color="#f093fb")
        ))
        
        fig_arima.update_layout(
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
            legend=dict(bgcolor='rgba(26,26,46,0.8)', bordercolor='#00ff88', borderwidth=2)
        )
        
        st.plotly_chart(fig_arima, config={'displayModeBar': True, 'responsive': True})
        
        # Display MSE
        col_mse1, col_mse2, col_mse3 = st.columns([1,2,1])
        with col_mse2:
            st.metric("📉 Mean Squared Error (MSE)", f"{arima_results['mse']:,.2f}")
        
        # Display forecast table
        st.markdown("#### 📊 Forecast Data")
        st.dataframe(arima_results['forecast_series'], width='stretch')
        
        # Display accuracy metrics
        accuracy = arima_results['accuracy_metrics']
        st.markdown("""
        <div class="accuracy-container">
            <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">📊 ARIMA Accuracy Metrics</h3>
        """, unsafe_allow_html=True)
        
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        
        with col_a1:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>MAPE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['mape']:.2%}</span>
            </div>
            <div class="accuracy-item">
                <strong>MAE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['mae']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a2:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>ME</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['me']:.2f}</span>
            </div>
            <div class="accuracy-item">
                <strong>MPE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['mpe']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a3:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>RMSE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['rmse']:.2f}</span>
            </div>
            <div class="accuracy-item">
                <strong>ACF1</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['acf1']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a4:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>Correlation</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['corr']:.2f}</span>
            </div>
            <div class="accuracy-item">
                <strong>Min-Max Error</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy['minmax']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ==================== SARIMA PREDICTION ====================
    st.markdown("""
    <div class="section-header">
        <h2>🎯 SARIMA Prediction Model</h2>
    </div>
    """, unsafe_allow_html=True)
    
    sarima_results = run_sarima_model(series, p, d, q, steps_forecast, rule)
    
    if sarima_results:
        # Display SARIMA forecast chart
        fig_sarima = go.Figure()
        fig_sarima.add_trace(go.Scatter(
            x=series.index, y=series,
            mode="lines", name="Actual Price", 
            line=dict(color="#667eea", width=3)
        ))
        fig_sarima.add_trace(go.Scatter(
            x=sarima_results['forecast_series'].index, 
            y=sarima_results['forecast_series'],
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
            legend=dict(bgcolor='rgba(26,26,46,0.8)', bordercolor='#00ff88', borderwidth=2)
        )
        
        st.plotly_chart(fig_sarima, config={'displayModeBar': True, 'responsive': True})
        
        # Display forecast table
        st.markdown("#### 📊 Forecast Data (SARIMA)")
        st.dataframe(sarima_results['forecast_series'], width='stretch')
        
        # Display accuracy metrics
        accuracy_sarima = sarima_results['accuracy_metrics']
        st.markdown("""
        <div class="accuracy-container">
            <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">📊 SARIMA Accuracy Metrics</h3>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>MAPE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['mape']:.2%}</span>
            </div>
            <div class="accuracy-item">
                <strong>MAE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['mae']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>ME</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['me']:.2f}</span>
            </div>
            <div class="accuracy-item">
                <strong>MPE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['mpe']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>RMSE</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['rmse']:.2f}</span>
            </div>
            <div class="accuracy-item">
                <strong>ACF1</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['acf1']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s4:
            st.markdown(f"""
            <div class="accuracy-item">
                <strong>Correlation</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['corr']:.2f}</span>
            </div>
            <div class="accuracy-item">
                <strong>Min-Max Error</strong><br>
                <span style="font-size: 1.5rem; color: #00ff88;">{accuracy_sarima['minmax']:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display MSE
        if 'mse' in sarima_results:
            col_mse_s1, col_mse_s2, col_mse_s3 = st.columns([1, 2, 1])
            with col_mse_s2:
                st.metric("📉 SARIMA Mean Squared Error (MSE)", f"{sarima_results['mse']:,.2f}")

else:
    st.error("❌ CoinGecko API did not return price data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6); padding: 2rem;">
    <p>🚀 Built with Streamlit | 📊 Data from CoinGecko API | 🤖 Powered by ARIMA & SARIMA Models</p>
</div>
""", unsafe_allow_html=True)