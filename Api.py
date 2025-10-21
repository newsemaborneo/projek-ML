import requests
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# ===================== PAGE CONFIG & CUSTOM CSS =====================
st.set_page_config(page_title="🚀 Bitcoin Forecast", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk styling modern
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(90deg, #f7931a 0%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(247, 147, 26, 0.3);
    }
    
    h2, h3 {
        color: #ffd700;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border-right: 2px solid #f7931a;
    }
    
    .css-1d391kg h2, [data-testid="stSidebar"] h2 {
        color: #f7931a;
        font-weight: 600;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7931a 0%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem;
        color: #ffffff;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f7d 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #f7931a;
        box-shadow: 0 8px 32px rgba(247, 147, 26, 0.2);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #f7931a 0%, #ffd700 100%);
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(247, 147, 26, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(247, 147, 26, 0.5);
    }
    
    /* Select Box Styling */
    .stSelectbox {
        background: rgba(30, 58, 95, 0.5);
        border-radius: 10px;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f7d 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #f7931a;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Accuracy Metrics Box */
    .metric-box {
        background: linear-gradient(135deg, #16213e 0%, #1e3a5f 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #f7931a;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(247, 147, 26, 0.1);
    }
    
    /* Glowing Effect */
    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 5px #f7931a, 0 0 10px #f7931a;
        }
        to {
            text-shadow: 0 0 10px #f7931a, 0 0 20px #f7931a, 0 0 30px #f7931a;
        }
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: #f7931a !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header dengan emoji dan styling
st.markdown("<h1 class='glow'>🚀 BITCOIN FORECAST DASHBOARD 📈</h1>", unsafe_allow_html=True)

# Auto refresh tiap 30 detik
st_autorefresh(interval=30 * 1000, key="btc_refresh")

# ===================== SIDEBAR =====================
st.sidebar.markdown("## ⚙️ Konfigurasi Model")
st.sidebar.markdown("---")

# Pilihan interval (HARUS DI ATAS!)
st.sidebar.markdown("### ⏰ Interval Waktu")
intervals = ["15 Menit", "30 Menit", "1 Jam", "1 Hari", "1 Bulan"]
choice = st.sidebar.selectbox("Pilih Interval", intervals)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Parameter ARIMA")
p = st.sidebar.number_input("🔵 p (AR term)", min_value=0, max_value=10, value=5, step=1, 
                            help="Autoregressive term - jumlah lag observation")
d = st.sidebar.number_input("🟢 d (Difference)", min_value=0, max_value=5, value=1, step=1,
                            help="Degree of differencing")
q = st.sidebar.number_input("🟡 q (MA term)", min_value=0, max_value=10, value=0, step=1,
                            help="Moving average term")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Pengaturan Prediksi")
steps_forecast = st.sidebar.slider("📈 Jumlah Langkah Prediksi", 
                                   min_value=5, max_value=200, value=30, step=5)

# Info box di sidebar
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips**: Gunakan p=5, d=1, q=0 untuk hasil optimal pada data Bitcoin")
st.sidebar.success("🔄 Dashboard auto-refresh setiap 30 detik")

# Atur days & resample rule sesuai interval
# Disesuaikan agar tidak terlalu banyak data points yang membuat model lambat
if choice == "15 Menit":
    days = "7"  # 7 hari untuk interval 15 menit
    rule = "15min"
elif choice == "30 Menit":
    days = "14"  # 14 hari untuk interval 30 menit
    rule = "30min"
elif choice == "1 Jam":
    days = "30"  # 30 hari untuk interval 1 jam
    rule = "1H"
elif choice == "1 Hari":
    days = "365"  # 1 tahun untuk interval harian
    rule = "4H"
elif choice == "1 Bulan":
    days = "365"  # 1 tahun untuk interval bulanan
    rule = "1D"

# ===================== FETCH DATA =====================
with st.spinner("🔄 Mengambil data dari CoinGecko..."):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    try:
        res = requests.get(url, params=params, timeout=30)
        data = res.json()
    except Exception as e:
        st.error(f"❌ Gagal mengambil data: {e}")
        st.stop()

if "prices" in data:
    # Konversi ke DataFrame dan ubah ke timezone WIB
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")

    # Resample ke OHLC
    ohlc = df.resample(rule, on="timestamp").ohlc().dropna()

    # ===================== METRIC CARDS =====================
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['price'].iloc[-1]
    price_change = ((current_price - df['price'].iloc[0]) / df['price'].iloc[0]) * 100
    high_price = ohlc["price"]["high"].max()
    low_price = ohlc["price"]["low"].min()
    
    with col1:
        st.metric("💰 Harga Saat Ini", f"${current_price:,.2f}", 
                 delta=f"{price_change:+.2f}%")
    
    with col2:
        st.metric("📈 Harga Tertinggi", f"${high_price:,.2f}")
    
    with col3:
        st.metric("📉 Harga Terendah", f"${low_price:,.2f}")
    
    with col4:
        st.metric("📊 Total Candles", f"{len(ohlc)}")

    st.markdown("---")

    # ===================== CANDLESTICK CHART =====================
    st.markdown("## 📊 Grafik Candlestick Real-Time")
    
    # Tentukan jumlah candle yang ditampilkan berdasarkan interval
    if choice == "15 Menit":
        display_candles = 96  # 1 hari (24 jam * 4 candle per jam)
    elif choice == "30 Menit":
        display_candles = 48  # 1 hari (24 jam * 2 candle per jam)
    elif choice == "1 Jam":
        display_candles = 72  # 3 hari (72 jam)
    elif choice == "1 Hari":
        display_candles = 90  # 15 hari (90 candle dengan interval 4 jam)
    elif choice == "1 Bulan":
        display_candles = 30  # 30 hari
    
    # Ambil data candle terbaru sesuai jumlah yang ditentukan
    ohlc_display = ohlc.tail(display_candles) if len(ohlc) > display_candles else ohlc
    
    fig = go.Figure(data=[go.Candlestick(
        x=ohlc_display.index,
        open=ohlc_display["price"]["open"],
        high=ohlc_display["price"]["high"],
        low=ohlc_display["price"]["low"],
        close=ohlc_display["price"]["close"],
        name="BTC/USD",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
        increasing_fillcolor="#00ff88",
        decreasing_fillcolor="#ff4444",
        increasing_line_width=2,
        decreasing_line_width=2
    )])

    chart_width = 2500 if choice != "1 Hari" else 1500
    fig.update_layout(
        title=dict(
            text=f"<b>BTC/USD Candlestick Chart - {choice} (Last {len(ohlc_display)} Candles)</b>",
            font=dict(size=24, color="#ffd700"),
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Waktu (WIB)",
        yaxis_title="Harga (USD)",
        template="plotly_dark",
        width=chart_width,
        height=900,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        hovermode="x unified"
    )

    fig.update_xaxes(rangeslider_visible=False, fixedrange=False, showgrid=True, gridcolor="#2a2a3e")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a3e")

    st.plotly_chart(fig, use_container_width=True)

    # ===================== CLOSING PRICE CHART =====================
    st.markdown("## 📈 Grafik Harga Penutupan")
    
    plt.style.use('dark_background')
    fig_close, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ohlc.index, ohlc["price"]["close"], label="Harga Penutupan", 
            color="#f7931a", linewidth=2.5, alpha=0.9)
    ax.fill_between(ohlc.index, ohlc["price"]["close"], alpha=0.3, color="#f7931a")
    ax.set_title("Grafik Harga Penutupan BTC/USD", color="#ffd700", fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Waktu (WIB)", color="#ffffff", fontsize=12)
    ax.set_ylabel("Harga (USD)", color="#ffffff", fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#f7931a", fontsize=10)
    ax.grid(color="#2a2a3e", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_facecolor("#0a0a0a")
    fig_close.patch.set_facecolor("#1a1a2e")
    st.pyplot(fig_close)
    plt.close()

    # ===================== DATASET OHLC =====================
    st.markdown("## 📑 Dataset OHLC (Open-High-Low-Close)")
    st.dataframe(ohlc, use_container_width=True, height=400)

    st.markdown("---")

    # ==================== ARIMA PREDICTION ====================
    st.markdown("## 🔮 Prediksi Menggunakan Model ARIMA")
    
    series = ohlc["price"]["close"]

    try:
        with st.spinner("🧮 Melatih model ARIMA..."):
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
        
            # Forecast dan buat index waktu yang benar
            forecast = model_fit.forecast(steps=steps_forecast)
            
            # Buat future index dengan benar (tanpa overlap dengan data terakhir)
            last_date = series.index[-1]
            future_index = pd.date_range(start=last_date + pd.Timedelta(rule), periods=steps_forecast, freq=rule)
            
            # Konversi forecast ke Series dengan index yang benar
            forecast_series = pd.Series(data=forecast.values if hasattr(forecast, 'values') else forecast, 
                                       index=future_index, 
                                       name='Forecast')

        # Plot prediksi ARIMA
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=series.index, y=series,
            mode="lines", name="Harga Aktual", 
            line=dict(color="#00d4ff", width=2.5)
        ))
        fig_pred.add_trace(go.Scatter(
            x=forecast_series.index, y=forecast_series,
            mode="lines+markers", name="Prediksi ARIMA", 
            line=dict(color="#ff9500", width=2.5, dash="dash"),
            marker=dict(size=6, color="#ff9500")
        ))

        fig_pred.update_layout(
            title=dict(
                text=f"<b>Prediksi Harga BTC/USD (ARIMA {p},{d},{q}) - {steps_forecast} Langkah</b>",
                font=dict(size=22, color="#ffd700"),
                x=0.5,
                xanchor="center"
            ),
            xaxis_title="Waktu",
            yaxis_title="Harga (USD)",
            template="plotly_dark",
            width=1500,
            height=600,
            plot_bgcolor="#0a0a0a",
            paper_bgcolor="#1a1a2e",
            hovermode="x unified"
        )
        fig_pred.update_xaxes(showgrid=True, gridcolor="#2a2a3e")
        fig_pred.update_yaxes(showgrid=True, gridcolor="#2a2a3e")

        st.plotly_chart(fig_pred, use_container_width=True)

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
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("📉 Mean Squared Error", f"{mse:,.2f}")

        # Tabel prediksi ARIMA
        st.markdown("### 📋 Data Prediksi ARIMA")
        st.dataframe(forecast_series, use_container_width=True)

        # Metrik akurasi ARIMA
        accuracy_metrics = forecast_accuracy(predictions, test.values)
        
        st.markdown("### 📊 Metrik Akurasi Model ARIMA")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>MAPE</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00ff88;'>{accuracy_metrics['mape']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>MAE</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00ff88;'>{accuracy_metrics['mae']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>RMSE</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00ff88;'>{accuracy_metrics['rmse']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>Correlation</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00ff88;'>{accuracy_metrics['corr']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Metrik tambahan
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>ME</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00d4ff;'>{accuracy_metrics['me']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>MPE</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00d4ff;'>{accuracy_metrics['mpe']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>ACF1</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00d4ff;'>{accuracy_metrics['acf1']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class='metric-box'>
                <h4 style='color: #f7931a; margin: 0;'>Min-Max Error</h4>
                <p style='font-size: 1.5rem; margin: 5px 0; color: #00d4ff;'>{accuracy_metrics['minmax']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ==================== SARIMA PREDICTION ====================
        st.markdown("## 🌟 Prediksi Menggunakan Model SARIMA")
        
        try:
            with st.spinner("🧮 Melatih model SARIMA..."):
                model_sarima = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                model_sarima_fit = model_sarima.fit(disp=False)
                sarima_forecast = model_sarima_fit.forecast(steps=steps_forecast)
                
                # Konversi forecast ke Series dengan index yang benar
                sarima_forecast_series = pd.Series(data=sarima_forecast.values if hasattr(sarima_forecast, 'values') else sarima_forecast,
                                                  index=future_index,
                                                  name='Forecast SARIMA')
       
            # Plot prediksi SARIMA
            fig_sarima = go.Figure()
            fig_sarima.add_trace(go.Scatter(
                x=series.index, y=series,
                mode="lines", name="Harga Aktual", 
                line=dict(color="#00d4ff", width=2.5)
            ))
            fig_sarima.add_trace(go.Scatter(
                x=sarima_forecast_series.index, y=sarima_forecast_series,
                mode="lines+markers", name="Prediksi SARIMA", 
                line=dict(color="#00ff88", width=2.5, dash="dash"),
                marker=dict(size=6, color="#00ff88")
            ))

            fig_sarima.update_layout(
                title=dict(
                    text=f"<b>Prediksi Harga BTC/USD (SARIMA {p},{d},{q}(1,1,1,12)) - {steps_forecast} Langkah</b>",
                    font=dict(size=22, color="#ffd700"),
                    x=0.5,
                    xanchor="center"
                ),
                xaxis_title="Waktu",
                yaxis_title="Harga (USD)",
                template="plotly_dark",
                width=1500,
                height=600,
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#1a1a2e",
                hovermode="x unified"
            )
            fig_sarima.update_xaxes(showgrid=True, gridcolor="#2a2a3e")
            fig_sarima.update_yaxes(showgrid=True, gridcolor="#2a2a3e")

            st.plotly_chart(fig_sarima, use_container_width=True)

            # Tabel prediksi SARIMA
            st.markdown("### 📋 Data Prediksi SARIMA")
            st.dataframe(sarima_forecast_series, use_container_width=True)

            # Metrik akurasi SARIMA
            accuracy_metrics_sarima = forecast_accuracy(model_sarima_fit.fittedvalues[train_size:], test.values)
            
            st.markdown("### 📊 Metrik Akurasi Model SARIMA")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>MAPE</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ffd700;'>{accuracy_metrics_sarima['mape']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>MAE</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ffd700;'>{accuracy_metrics_sarima['mae']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>RMSE</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ffd700;'>{accuracy_metrics_sarima['rmse']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>Correlation</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ffd700;'>{accuracy_metrics_sarima['corr']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrik tambahan SARIMA
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>ME</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ff9500;'>{accuracy_metrics_sarima['me']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>MPE</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ff9500;'>{accuracy_metrics_sarima['mpe']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>ACF1</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ff9500;'>{accuracy_metrics_sarima['acf1']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: #00ff88; margin: 0;'>Min-Max Error</h4>
                    <p style='font-size: 1.5rem; margin: 5px 0; color: #ff9500;'>{accuracy_metrics_sarima['minmax']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Gagal menjalankan SARIMA: {e}")

    except Exception as e:
        st.error(f"❌ Gagal menjalankan ARIMA: {e}")

else:
    st.error("❌ API CoinGecko tidak mengembalikan data harga.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>📊 <b>Bitcoin Forecast Dashboard</b> | Powered by CoinGecko API</p>
        <p>⚡ Real-time data with 30-second auto-refresh</p>
    </div>
""", unsafe_allow_html=True)