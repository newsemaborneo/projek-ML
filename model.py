import requests
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


def fetch_bitcoin_data(days):
    """Fetch Bitcoin price data from CoinGecko API"""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    
    try:
        res = requests.get(url, params=params, timeout=30)
        return res.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        st.stop()


def prepare_ohlc_data(data, rule):
    """Prepare OHLC data from raw price data"""
    # Convert to DataFrame and change timezone to WIB
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")
    
    # Resample to OHLC
    ohlc = df.resample(rule, on="timestamp").ohlc().dropna()
    
    return df, ohlc


def perform_adf_test(timeseries, d):
    """
    Perform Augmented Dickey-Fuller test and display results
    """
    result = adfuller(timeseries, autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    is_stationary = p_value < 0.05
    
    # Display original series test
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
                {adf_stat:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_adf2:
        p_color = "#00ff88" if p_value < 0.05 else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                    border-radius: 10px; backdrop-filter: blur(10px);">
            <h4 style="color: white; margin: 0;">p-value</h4>
            <p style="font-size: 2rem; color: {p_color}; font-weight: bold; margin: 0.5rem 0;">
                {p_value:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_adf3:
        status = "✅ STATIONARY" if is_stationary else "⚠️ NON-STATIONARY"
        status_color = "#00ff88" if is_stationary else "#ff0066"
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
        cv_1_color = "#00ff88" if adf_stat < critical_values['1%'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; 
                    border-radius: 8px; text-align: center;">
            <strong style="color: white;">1% Level</strong><br>
            <span style="font-size: 1.3rem; color: {cv_1_color};">
                {critical_values['1%']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_cv2:
        cv_5_color = "#00ff88" if adf_stat < critical_values['5%'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; 
                    border-radius: 8px; text-align: center;">
            <strong style="color: white;">5% Level</strong><br>
            <span style="font-size: 1.3rem; color: {cv_5_color};">
                {critical_values['5%']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_cv3:
        cv_10_color = "#00ff88" if adf_stat < critical_values['10%'] else "#ff0066"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; 
                    border-radius: 8px; text-align: center;">
            <strong style="color: white;">10% Level</strong><br>
            <span style="font-size: 1.3rem; color: {cv_10_color};">
                {critical_values['10%']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Interpretation
    if is_stationary:
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
        series_diff = timeseries.diff(d).dropna()
        result_diff = adfuller(series_diff, autolag='AIC')
        
        adf_stat_diff = result_diff[0]
        p_value_diff = result_diff[1]
        is_stationary_diff = p_value_diff < 0.05
        
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
                    {adf_stat_diff:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_diff2:
            p_color_diff = "#00ff88" if p_value_diff < 0.05 else "#ff0066"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4 style="color: white; margin: 0;">p-value</h4>
                <p style="font-size: 2rem; color: {p_color_diff}; font-weight: bold; margin: 0.5rem 0;">
                    {p_value_diff:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_diff3:
            status_diff = "✅ STATIONARY" if is_stationary_diff else "⚠️ NON-STATIONARY"
            status_color_diff = "#00ff88" if is_stationary_diff else "#ff0066"
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
        
        if is_stationary_diff:
            st.success(f"""
            ✅ **Data is now STATIONARY after {d} differencing!** - Ready for ARIMA modeling.
            """)
        else:
            st.warning(f"""
            ⚠️ **Data is still NON-STATIONARY after {d} differencing** - Consider increasing 'd' parameter.
            """)
    
    st.markdown("---")
    
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }


def forecast_accuracy(forecast, actual):
    """Calculate various forecast accuracy metrics"""
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


def run_arima_model(series, p, d, q, steps_forecast, rule):
    """Run ARIMA model and return results"""
    try:
        with st.spinner("🤖 Training ARIMA model..."):
            # Train model
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=steps_forecast)
            future_index = pd.date_range(start=series.index[-1], periods=steps_forecast+1, freq=rule)[1:]
            forecast_series = pd.Series(forecast, index=future_index)
            
            # Evaluate model
            train_size = int(len(series) * 0.8)
            train, test = series[:train_size], series[train_size:]
            model_eval = ARIMA(train, order=(p, d, q)).fit()
            predictions = model_eval.forecast(steps=len(test))
            mse = mean_squared_error(test, predictions)
            
            # Calculate accuracy metrics
            accuracy_metrics = forecast_accuracy(predictions, test.values)
            
            return {
                'forecast_series': forecast_series,
                'mse': mse,
                'accuracy_metrics': accuracy_metrics
            }
    except Exception as e:
        st.error(f"❌ ARIMA Error: {e}")
        return None


def run_sarima_model(series, p, d, q, steps_forecast, rule):
    """Run SARIMA model and return results"""
    try:
        with st.spinner("🤖 Training SARIMA model..."):
            # Train SARIMA model
            model_sarima = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
            model_sarima_fit = model_sarima.fit(disp=False)
            
            # Generate forecast
            sarima_forecast = model_sarima_fit.forecast(steps=steps_forecast)
            future_index = pd.date_range(start=series.index[-1], periods=steps_forecast+1, freq=rule)[1:]
            sarima_forecast_series = pd.Series(sarima_forecast, index=future_index)
            
            # Calculate accuracy metrics
            train_size = int(len(series) * 0.8)
            train, test = series[:train_size], series[train_size:]
            accuracy_metrics_sarima = forecast_accuracy(model_sarima_fit.fittedvalues[train_size:], test.values)
            
            # Calculate MSE
            try:
                mse_sarima = mean_squared_error(test, sarima_forecast[:len(test)])
            except:
                mse_sarima = None
            
            return {
                'forecast_series': sarima_forecast_series,
                'mse': mse_sarima,
                'accuracy_metrics': accuracy_metrics_sarima
            }
    except Exception as e:
        st.error(f"❌ SARIMA Error: {e}")
        return None