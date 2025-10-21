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
st.set_page_config(page_title="📊 Forecast Bitcoin", layout="wide")

# Custom CSS untuk styling ultra modern
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(135deg, #f7931a 0%, #ffd700 50%, #f7931a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        text-align: center;
        font-size: 4rem !important;
        margin-bottom: 2rem;
        text-shadow: 0 0 40px rgba(247, 147, 26, 0.5);
        letter-spacing: 3px;
        animation: glow-pulse 3s ease-in-out infinite;
    }
    
    @keyframes glow-pulse {
        0%, 100% { 
            filter: drop-shadow(0 0 10px rgba(247, 147, 26, 0.7));
        }
        50% { 
            filter: drop-shadow(0 0 30px rgba(255, 215, 0, 0.9)) drop-shadow(0 0 50px rgba(247, 147, 26, 0.6));
        }
    }
    
    h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        color: #ffd700;
        font-weight: 700;
        margin-top: 2rem;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
        letter-spacing: 1.5px;
    }
    
    /* Sidebar Ultra Modern */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f3a 50%, #16213e 100%);
        border-right: 3px solid #f7931a;
        box-shadow: 5px 0 30px rgba(247, 147, 26, 0.3);
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #f7931a;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(247, 147, 26, 0.6);
    }
    
    /* Metric Cards - Premium Style */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #f7931a 0%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(247, 147, 26, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.3rem;
        color: #ffffff;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.4) 0%, rgba(44, 95, 125, 0.4) 100%);
        padding: 25px;
        border-radius: 20px;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(135deg, #f7931a, #ffd700, #f7931a);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        z-index: -1;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(247, 147, 26, 0.4);
    }
    
    /* Dataframe Premium */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(247, 147, 26, 0.3);
    }
    
    /* Button Futuristic */
    .stButton>button {
        background: linear-gradient(135deg, #f7931a 0%, #ffd700 100%);
        color: #000;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 12px 35px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(247, 147, 26, 0.4);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(247, 147, 26, 0.6);
        background: linear-gradient(135deg, #ffd700 0%, #f7931a 100%);
    }
    
    /* Select Box Modern */
    .stSelectbox {
        background: rgba(30, 58, 95, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Number Input & Slider Modern */
    .stNumberInput, .stSlider {
        background: rgba(30, 58, 95, 0.2);
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Info/Success Box */
    .stAlert {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.5) 0%, rgba(44, 95, 125, 0.5) 100%);
        border-left: 5px solid #f7931a;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Scrollbar Custom */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e27;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #f7931a, #ffd700);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ffd700, #f7931a);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #f7931a !important;
        border-right-color: #ffd700 !important;
    }
    
    /* Subheader Style */
    .stSubheader {
        color: #ffd700;
        font-weight: 700;
        font-size: 1.8rem;
        margin-top: 2rem;
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.4);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Text Color */
    p, span, div {
        color: #e0e0e0;
    }
    
    /* Premium Glow Effect */
    .premium-glow {
        animation: premium-glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes premium-glow {
        from {
            filter: drop-shadow(0 0 5px rgba(247, 147, 26, 0.5));
        }
        to {
            filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.8));
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header Ultra Modern
st.markdown("<h1 class='premium-glow'>📊 FORECAST BITCOIN (BTC/USD)</h1>", unsafe_allow_html=True)

# Auto refresh tiap 30 detik
st_autorefresh(interval=30 * 1000, key="btc_refresh")

# ===================== Sidebar =====================
st.sidebar.markdown("## ⚙️ Forecast Bitcoin")
st.sidebar.markdown("---")

p = st.sidebar.number_input("p (AR term)", min_value=0, max_value=10, value=5, step=1)
d = st.sidebar.number_input("d (Difference)", min_value=0, max_value=5, value=1, step=1)
q = st.sidebar.number_input("q (MA term)", min_value=0, max_value=10, value=0, step=1)

steps_forecast = st.sidebar.slider("📈 Jumlah langkah prediksi ke depan", 
                                   min_value=5, max_value=200, value=30, step=5)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip**: Parameter p=5, d=1, q=0 optimal untuk Bitcoin")
st.sidebar.success("🔄 Auto-refresh aktif setiap 30 detik")

# Pilihan interval
intervals = ["15 Menit", "30 Menit", "1 Jam", "1 Hari", "1 Bulan"]
choice = st.selectbox("Pilih Interval", intervals)

# Atur days & resample rule sesuai interval
if choice in ["15 Menit", "30 Menit", "1 Jam"]:
    days = "1"  # ambil data 1 hari
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

st.markdown("---")

# Panggil API CoinGecko
with st.spinner("🔄 Mengambil data real-time dari CoinGecko..."):
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

    # Harga terakhir - Premium Display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("💰 Harga Terakhir BTC/USD", f"${df['price'].iloc[-1]:,.2f}", 
                 delta=f"{((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0] * 100):+.2f}%")

    st.markdown("---")

    # Candlestick chart
    st.markdown("## 📊 Grafik Candlestick Real-Time")
    
    fig = go.Figure(data=[go.Candlestick(
        x=ohlc.index,
        open=ohlc["price"]["open"],
        high=ohlc["price"]["high"],
        low=ohlc["price"]["low"],
        close=ohlc["price"]["close"],
        name="BTC/USD",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
        increasing_fillcolor="rgba(0, 255, 136, 0.3)",
        decreasing_fillcolor="rgba(255, 68, 68, 0.3)",
        increasing_line_width=2.5,
        decreasing_line_width=2.5
    )])

    # Layout chart - Ultra Modern
    chart_width = 2500 if choice != "1 Hari" else 1500
    fig.update_layout(
        title=dict(
            text=f"<b>BTC/USD - Candlestick ({choice})</b>",
            font=dict(size=26, color="#ffd700", family="Rajdhani"),
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Waktu (WIB)",
        yaxis_title="Harga (USD)",
        template="plotly_dark",
        width=chart_width,
        height=900,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0a0e27",
        paper_bgcolor="#1a1f3a",
        font=dict(color="#e0e0e0", family="Rajdhani"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a1f3a",
            font_size=14,
            font_family="Rajdhani"
        )
    )

    # Zoom ke 100 candle terakhir (khusus 1 Hari)
    if choice == "1 Hari" and len(ohlc) > 100:
        fig.update_xaxes(range=[ohlc.index[-100], ohlc.index[-1]])

    # Scroll horizontal dengan grid modern
    fig.update_xaxes(
        rangeslider_visible=False, 
        fixedrange=False,
        showgrid=True,
        gridcolor="rgba(247, 147, 26, 0.1)",
        gridwidth=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(247, 147, 26, 0.1)",
        gridwidth=1
    )

    # Tampilkan chart
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Dataset OHLC dengan header modern
    st.markdown("## 📑 Dataset (OHLC)")
    st.dataframe(ohlc, use_container_width=True, height=400)

    st.markdown("---")

    # Membuat plot harga penutupan dengan style ultra modern
    st.markdown("## 📈 Grafik Harga Penutupan")
    
    plt.style.use('dark_background')
    fig_close, ax = plt.subplots(figsize=(14, 6), facecolor='#1a1f3a')
    ax.set_facecolor('#0a0e27')
    
    # Plot dengan gradient effect
    ax.plot(ohlc.index, ohlc["price"]["close"], 
            label="Harga Penutupan", 
            color="#f7931a", 
            linewidth=3, 
            alpha=0.9)
    ax.fill_between(ohlc.index, ohlc["price"]["close"], 
                     alpha=0.25, 
                     color="#f7931a")
    
    ax.set_title("Grafik Harga Penutupan BTC/USD", 
                 color="#ffd700", 
                 fontsize=20, 
                 fontweight="bold", 
                 pad=20,
                 family="sans-serif")
    ax.set_xlabel("Waktu (WIB)", color="#e0e0e0", fontsize=13, fontweight="bold")
    ax.set_ylabel("Harga (USD)", color="#e0e0e0", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1f3a", 
              edgecolor="#f7931a", 
              fontsize=11,
              framealpha=0.9)
    ax.grid(color="#f7931a", linestyle="--", linewidth=0.5, alpha=0.2)
    ax.tick_params(colors="#e0e0e0")
    
    st.pyplot(fig_close)
    plt.close()

    st.markdown("---")

    # ==================== 🔮 Prediksi ARIMA ====================
    st.markdown("## 🔮 Prediksi Harga BTC/USD")

    # Gunakan harga penutupan
    series = ohlc["price"]["close"]

    try:
        with st.spinner("🧮 Training Model ARIMA..."):
            # Latih model ARIMA dengan parameter dari sidebar
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
        
            # Forecast sesuai jumlah langkah yang dipilih user
            forecast = model_fit.forecast(steps=steps_forecast)

            # Buat index waktu untuk prediksi
            future_index = pd.date_range(start=series.index[-1], periods=steps_forecast+1, freq=rule)[1:]
            forecast_series = pd.Series(forecast, index=future_index)

        # Plot prediksi dengan style ultra modern
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=series.index, 
            y=series,
            mode="lines", 
            name="Harga Aktual", 
            line=dict(color="#00d4ff", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.1)"
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=forecast_series.index, 
            y=forecast_series,
            mode="lines+markers", 
            name="Prediksi ARIMA", 
            line=dict(color="#ff9500", width=3, dash="dash"),
            marker=dict(size=7, color="#ff9500", symbol="diamond")
        ))

        fig_pred.update_layout(
            title=dict(
                text=f"<b>Prediksi Harga BTC/USD (ARIMA {p},{d},{q}) - {steps_forecast} langkah ke depan</b>",
                font=dict(size=24, color="#ffd700", family="Rajdhani"),
                x=0.5,
                xanchor="center"
            ),
            xaxis_title="Waktu",
            yaxis_title="Harga (USD)",
            template="plotly_dark",
            width=1500,
            height=600,
            plot_bgcolor="#0a0e27",
            paper_bgcolor="#1a1f3a",
            font=dict(color="#e0e0e0", family="Rajdhani"),
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                bgcolor="rgba(26, 31, 58, 0.8)",
                bordercolor="#f7931a",
                borderwidth=2
            )
        )
        
        fig_pred.update_xaxes(showgrid=True, gridcolor="rgba(247, 147, 26, 0.1)")
        fig_pred.update_yaxes(showgrid=True, gridcolor="rgba(247, 147, 26, 0.1)")

        st.plotly_chart(fig_pred, use_container_width=True)

        def forecast_accuracy(forecast, actual):
            # Konversi ke array NumPy jika belum
            forecast = np.array(forecast)
            actual = np.array(actual)

            mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
            me = np.mean(forecast - actual)                            # ME
            mae = np.mean(np.abs(forecast - actual))                   # MAE
            mpe = np.mean((forecast - actual) / actual)                # MPE
            rmse = np.mean((forecast - actual) ** 2) ** 0.5            # RMSE
            corr = np.corrcoef(forecast, actual)[0, 1]                 # Correlation

            # Perbaikan untuk multi-dimensional indexing
            mins = np.amin(np.column_stack([forecast, actual]), axis=1)
            maxs = np.amax(np.column_stack([forecast, actual]), axis=1)
            minmax = 1 - np.mean(mins / maxs)                          # Min-Max Error

            acf1 = acf(forecast - actual)[1]                           # ACF1
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

        # MSE Metric Premium
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("📉 Mean Squared Error (MSE)", f"{mse:,.2f}")

        st.markdown("---")

        # Tabel prediksi
        st.markdown("### 📑 Data Prediksi ke Depan")
        st.dataframe(forecast_series, use_container_width=True)

        # Hitung metrik akurasi
        accuracy_metrics = forecast_accuracy(predictions, test.values)

        # Tampilkan metrik akurasi di Streamlit dengan style modern
        st.markdown("### 📊 Metrik Akurasi Prediksi ARIMA")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(30, 58, 95, 0.4), rgba(44, 95, 125, 0.4)); 
                        padding: 20px; border-radius: 15px; border: 2px solid #f7931a; 
                        box-shadow: 0 4px 20px rgba(247, 147, 26, 0.2); margin: 10px 0;'>
                <h4 style='color: #ffd700; margin: 0; font-family: Rajdhani;'>📊 Error Metrics</h4>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>MAPE:</b> {accuracy_metrics['mape']:.2%}</p>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>MAE:</b> {accuracy_metrics['mae']:.2f}</p>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>RMSE:</b> {accuracy_metrics['rmse']:.2f}</p>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>Min-Max Error:</b> {accuracy_metrics['minmax']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(30, 58, 95, 0.4), rgba(44, 95, 125, 0.4)); 
                        padding: 20px; border-radius: 15px; border: 2px solid #f7931a; 
                        box-shadow: 0 4px 20px rgba(247, 147, 26, 0.2); margin: 10px 0;'>
                <h4 style='color: #ffd700; margin: 0; font-family: Rajdhani;'>📈 Statistical Metrics</h4>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>ME:</b> {accuracy_metrics['me']:.2f}</p>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>MPE:</b> {accuracy_metrics['mpe']:.2%}</p>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>Correlation:</b> {accuracy_metrics['corr']:.2f}</p>
                <p style='color: #e0e0e0; margin: 10px 0;'><b>ACF1:</b> {accuracy_metrics['acf1']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # ==================== SARIMA ====================
        st.markdown("## 🌟 Prediksi menggunakan SARIMA")
        
        try:
            with st.spinner("🧮 Training Model SARIMA..."):
                model_sarima = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                model_sarima_fit = model_sarima.fit(disp=False)
                sarima_forecast = model_sarima_fit.forecast(steps=steps_forecast)
                sarima_forecast_series = pd.Series(sarima_forecast, index=future_index)
       
            # Plot prediksi SARIMA
            fig_sarima = go.Figure()
            
            fig_sarima.add_trace(go.Scatter(
                x=series.index, 
                y=series,
                mode="lines", 
                name="Harga Aktual", 
                line=dict(color="#00d4ff", width=3),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.1)"
            ))
            
            fig_sarima.add_trace(go.Scatter(
                x=sarima_forecast_series.index, 
                y=sarima_forecast_series,
                mode="lines+markers", 
                name="Prediksi SARIMA", 
                line=dict(color="#00ff88", width=3, dash="dash"),
                marker=dict(size=7, color="#00ff88", symbol="star")
            ))

            fig_sarima.update_layout(
                title=dict(
                    text=f"<b>Prediksi Harga BTC/USD (SARIMA {p},{d},{q}(1,1,1,12)) - {steps_forecast} langkah ke depan</b>",
                    font=dict(size=24, color="#ffd700", family="Rajdhani"),
                    x=0.5,
                    xanchor="center"
                ),
                xaxis_title="Waktu",
                yaxis_title="Harga (USD)",
                template="plotly_dark",
                width=1500,
                height=600,
                plot_bgcolor="#0a0e27",
                paper_bgcolor="#1a1f3a",
                font=dict(color="#e0e0e0", family="Rajdhani"),
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(26, 31, 58, 0.8)",
                    bordercolor="#00ff88",
                    borderwidth=2
                )
            )
            
            fig_sarima.update_xaxes(showgrid=True, gridcolor="rgba(0, 255, 136, 0.1)")
            fig_sarima.update_yaxes(showgrid=True, gridcolor="rgba(0, 255, 136, 0.1)")

            st.plotly_chart(fig_sarima, use_container_width=True)

            # Tabel prediksi SARIMA
            st.markdown("### 📑 Data Prediksi ke Depan (SARIMA)")
            st.dataframe(sarima_forecast_series, use_container_width=True)

            # Metrik akurasi SARIMA
            accuracy_metrics_sarima = forecast_accuracy(model_sarima_fit.fittedvalues[train_size:], test.values)
            
            st.markdown("### 📊 Metrik Akurasi Prediksi SARIMA")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(30, 58, 95, 0.4), rgba(44, 95, 125, 0.4)); 
                            padding: 20px; border-radius: 15px; border: 2px solid #00ff88; 
                            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2); margin: 10px 0;'>
                    <h4 style='color: #00ff88; margin: 0; font-family: Rajdhani;'>📊 Error Metrics</h4>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>MAPE:</b> {accuracy_metrics_sarima['mape']:.2%}</p>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>MAE:</b> {accuracy_metrics_sarima['mae']:.2f}</p>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>RMSE:</b> {accuracy_metrics_sarima['rmse']:.2f}</p>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>Min-Max Error:</b> {accuracy_metrics_sarima['minmax']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(30, 58, 95, 0.4), rgba(44, 95, 125, 0.4)); 
                            padding: 20px; border-radius: 15px; border: 2px solid #00ff88; 
                            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2); margin: 10px 0;'>
                    <h4 style='color: #00ff88; margin: 0; font-family: Rajdhani;'>📈 Statistical Metrics</h4>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>ME:</b> {accuracy_metrics_sarima['me']:.2f}</p>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>MPE:</b> {accuracy_metrics_sarima['mpe']:.2%}</p>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>Correlation:</b> {accuracy_metrics_sarima['corr']:.2f}</p>
                    <p style='color: #e0e0e0; margin: 10px 0;'><b>ACF1:</b> {accuracy_metrics_sarima['acf1']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Gagal menjalankan SARIMA: {e}")

    except Exception as e:
        st.error(f"❌ Gagal menjalankan ARIMA: {e}")

else:
    st.error("❌ API CoinGecko tidak mengembalikan data harga.")

# Footer Ultra Modern
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(30, 58, 95, 0.3), rgba(44, 95, 125, 0.3)); 
                border-radius: 20px; margin-top: 40px; border: 1px solid rgba(247, 147, 26, 0.3);'>
        <h3 style='color: #ffd700; font-family: Orbitron; margin-bottom: 15px; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);'>
            🚀 BITCOIN FORECAST DASHBOARD 🚀
        </h3>
        <p style='color: #e0e0e0; font-size: 1.1rem; font-family: Rajdhani; margin: 10px 0;'>
            ⚡ <b>Real-time data</b> powered by <b>CoinGecko API</b>
        </p>
        <p style='color: #f7931a; font-size: 1rem; font-family: Rajdhani; margin: 10px 0;'>
            🔄 Auto-refresh setiap 30 detik | 📊 ARIMA & SARIMA Forecasting
        </p>
        <p style='color: #888; font-size: 0.9rem; margin-top: 15px;'>
            Made with 💛 for Bitcoin Traders & Analysts
        </p>
    </div>
""", unsafe_allow_html=True)