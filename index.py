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


st.set_page_config(page_title="📊 Forecast Bitcoin", layout="wide")
st.title("📊 Forecast Bitcoin (BTC/USD)")

# Auto refresh tiap 30 detik
st_autorefresh(interval=30 * 1000, key="btc_refresh")

# ===================== Sidebar =====================
st.sidebar.header("⚙️ Forecast Bitcoin")
p = st.sidebar.number_input("p (AR term)", min_value=0, max_value=10, value=5, step=1)
d = st.sidebar.number_input("d (Difference)", min_value=0, max_value=5, value=1, step=1)
q = st.sidebar.number_input("q (MA term)", min_value=0, max_value=10, value=0, step=1)

steps_forecast = st.sidebar.slider("📈 Jumlah langkah prediksi ke depan", 
                                   min_value=5, max_value=200, value=30, step=5)

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


# Panggil API CoinGecko
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": days}
try:
    res = requests.get(url, params=params, timeout=30)
    data = res.json()
except Exception as e:
    st.error(f"Gagal mengambil data: {e}")
    st.stop()

if "prices" in data:
    # Konversi ke DataFrame dan ubah ke timezone WIB
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")

    # Resample ke OHLC
    ohlc = df.resample(rule, on="timestamp").ohlc().dropna()

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=ohlc.index,
        open=ohlc["price"]["open"],
        high=ohlc["price"]["high"],
        low=ohlc["price"]["low"],
        close=ohlc["price"]["close"],
        name="BTC/USD",
        increasing_line_color="green",
        decreasing_line_color="red",
        increasing_line_width=2,
        decreasing_line_width=2
    )])

    # Layout chart
    chart_width = 2500 if choice != "1 Hari" else 1500
    fig.update_layout(
        title=f"BTC/USD - Candlestick ({choice})",
        xaxis_title="Waktu (WIB)",
        yaxis_title="Harga (USD)",
        template="plotly_dark",
        width=chart_width,
        height=900,
        xaxis_rangeslider_visible=False
    )

    # Zoom ke 100 candle terakhir (khusus 1 Hari)
    if choice == "1 Hari" and len(ohlc) > 100:
        fig.update_xaxes(range=[ohlc.index[-100], ohlc.index[-1]])

    # Scroll horizontal
    fig.update_xaxes(rangeslider_visible=False, fixedrange=False)

    # Tampilkan chart
    st.plotly_chart(fig, use_container_width=True)

    # Harga terakhir
    st.metric("💰 Harga Terakhir BTC/USD", f"${df['price'].iloc[-1]:,.2f}")

    # Dataset OHLC
    st.subheader("📑 Dataset (OHLC)")
    st.dataframe(ohlc)

    # Membuat plot harga penutupan dengan template plotly_dark
    plt.style.use('dark_background')  # Mengatur gaya menjadi dark background
    plt.figure(figsize=(10, 5))
    plt.plot(ohlc.index, ohlc["price"]["close"], label="Harga Penutupan", color="cyan")  # Warna garis disesuaikan
    plt.title("Grafik Harga Penutupan BTC/USD", color="white")  # Warna judul
    plt.xlabel("Waktu (WIB)", color="white")  # Warna label x
    plt.ylabel("Harga (USD)", color="white")  # Warna label y
    plt.legend(facecolor="black", edgecolor="white")  # Warna latar belakang legend
    plt.grid(color="gray", linestyle="--", linewidth=0.5)  # Warna dan gaya grid
    st.pyplot(plt)

    # ==================== 🔮 Prediksi ARIMA ====================
    st.subheader("🔮 Prediksi Harga BTC/USD")

    # Gunakan harga penutupan
    series = ohlc["price"]["close"]

    try:
        # Latih model ARIMA dengan parameter dari sidebar
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
    
        # Forecast sesuai jumlah langkah yang dipilih user
        forecast = model_fit.forecast(steps=steps_forecast)

        # Buat index waktu untuk prediksi
        future_index = pd.date_range(start=series.index[-1], periods=steps_forecast+1, freq=rule)[1:]
        forecast_series = pd.Series(forecast, index=future_index)

        # Plot prediksi
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=series.index, y=series,
                                      mode="lines", name="Harga Aktual", line=dict(color="blue")))
        fig_pred.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series,
                                      mode="lines", name="Prediksi ARIMA", line=dict(color="orange", dash="dash")))

        fig_pred.update_layout(
            title=f"Prediksi Harga BTC/USD (ARIMA {p},{d},{q}) - {steps_forecast} langkah ke depan",
            xaxis_title="Waktu",
            yaxis_title="Harga (USD)",
            template="plotly_dark",
            width=1500,
            height=600
        )

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

        st.metric("📉 Mean Squared Error (MSE)", f"{mse:,.2f}")

        # Tabel prediksi
        st.subheader("📑 Data Prediksi ke Depan")
        st.dataframe(forecast_series)

        # Hitung metrik akurasi
        accuracy_metrics = forecast_accuracy(predictions, test.values)

        # Tampilkan metrik akurasi di Streamlit
        st.subheader("📊 Metrik Akurasi Prediksi ARIMA")
        st.write(f"Mean Absolute Percentage Error (MAPE): {accuracy_metrics['mape']:.2%}")
        st.write(f"Mean Error (ME): {accuracy_metrics['me']:.2f}")
        st.write(f"Mean Absolute Error (MAE): {accuracy_metrics['mae']:.2f}")
        st.write(f"Mean Percentage Error (MPE): {accuracy_metrics['mpe']:.2%}")
        st.write(f"Root Mean Squared Error (RMSE): {accuracy_metrics['rmse']:.2f}")
        st.write(f"Autocorrelation of Residuals (ACF1): {accuracy_metrics['acf1']:.2f}")
        st.write(f"Correlation Coefficient: {accuracy_metrics['corr']:.2f}")
        st.write(f"Min-Max Error: {accuracy_metrics['minmax']:.2%}")
        
        
        st.subheader("Prediksi mengunakan Sarima")
        try:
            model_sarima = SARIMAX(series , order=(p, d, q), seasonal_order=( 1, 1, 1, 12))
            model_sarima_fit = model_sarima.fit(disp=False)
            sarima_forecast = model_sarima_fit.forecast(steps=steps_forecast)
            sarima_forecast_series = pd.Series(sarima_forecast, index=future_index)
       
            # Plot prediksi SARIMA
            fig_sarima = go.Figure()
            fig_sarima.add_trace(go.Scatter(x=series.index, y=series,
                                          mode="lines", name="Harga Aktual", line=dict(color="blue")))
            fig_sarima.add_trace(go.Scatter(x=sarima_forecast_series.index, y=sarima_forecast_series,
                                          mode="lines", name="Prediksi SARIMA", line=dict(color="green", dash="dash")))

            fig_sarima.update_layout(
                title=f"Prediksi Harga BTC/USD (SARIMA {p},{d},{q}(1,1,1,12)) - {steps_forecast} langkah ke depan",
                xaxis_title="Waktu",
                yaxis_title="Harga (USD)",
                template="plotly_dark",
                width=1500,
                height=600
            )

            st.plotly_chart(fig_sarima, use_container_width=True)

            # Tabel prediksi SARIMA
            st.subheader("📑 Data Prediksi ke Depan (SARIMA)")
            st.dataframe(sarima_forecast_series)

            accuracy_metrics_sarima = forecast_accuracy(model_sarima_fit.fittedvalues[train_size:], test.values)  
            st.subheader("📊 Metrik Akurasi Prediksi SARIMA")
            st.write(f"Mean Absolute Percentage Error (MAPE): {accuracy_metrics_sarima['mape']:.2%}")   
            st.write(f"Mean Error (ME): {accuracy_metrics_sarima['me']:.2f}")   
            st.write(f"Mean Absolute Error (MAE): {accuracy_metrics_sarima['mae']:.2f}")   
            st.write(f"Mean Percentage Error (MPE): {accuracy_metrics_sarima['mpe']:.2%}")   
            st.write(f"Root Mean Squared Error (RMSE): {accuracy_metrics_sarima['rmse']:.2f}")  
            st.write(f"Autocorrelation of Residuals (ACF1): {accuracy_metrics_sarima['acf1']:.2f}")   
            st.write(f"Correlation Coefficient: {accuracy_metrics_sarima['corr']:.2f}")   
            st.write(f"Min-Max Error: {accuracy_metrics_sarima['minmax']:.2%}")  
        except Exception as e:
            st.error(f"Gagal menjalankan SARIMA: {e}")


    except Exception as e:
        st.error(f"Gagal menjalankan ARIMA: {e}")


else:
    st.error("API CoinGecko tidak mengembalikan data harga.")