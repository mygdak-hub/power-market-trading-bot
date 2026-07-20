import streamlit as st
import pandas as pd
import numpy as np
import requests

# Machine Learning Forecasting Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Power Market Simulator", layout="wide")

MARKET_NAMES = {
    "DE-LU": "🇩🇪🇱🇺 Germany / Luxembourg (EPEX SPOT)",
    "FR": "🇫🇷 France (EPEX SPOT)",
    "NL": "🇳🇱 Netherlands (EPEX SPOT)",
    "BE": "🇧🇪 Belgium (EPEX SPOT)",
    "AT": "🇦🇹 Austria (EXAA / EPEX SPOT)",
    "PL": "🇵🇱 Poland (TGE)",
    "DK1": "🇩🇰 Denmark West (Nord Pool)",
    "NO2": "🇳🇴 Norway South (Nord Pool)"
}

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Market & Model Settings")

selected_bzn = st.sidebar.selectbox(
    "Select Target Power Market",
    options=list(MARKET_NAMES.keys()),
    format_func=lambda x: MARKET_NAMES[x],
    index=0
)

selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    options=[
        "Actual Spot Price (Fraunhofer ISE)",
        "LightGBM Forecast",
        "XGBoost Forecast",
        "CatBoost Forecast",
        "Random Forest Forecast",
        "24h Moving Average Trend"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Strategy Parameters")
buy_threshold = st.sidebar.slider("Charge Trigger Price (€/MWh)", min_value=0, max_value=60, value=35)
sell_threshold = st.sidebar.slider("Discharge Trigger Price (€/MWh)", min_value=60, max_value=200, value=90)


# --- DATA FETCHING ---
@st.cache_data(ttl=1800)
def fetch_fraunhofer_prices(zone):
    url = "https://api.energy-charts.info/price"
    params = {"bzn": zone}
    res = requests.get(url, params=params).json()
    return pd.DataFrame({
        "Timestamp": pd.to_datetime(res["unix_seconds"], unit="s"),
        "Day-Ahead Price (€/MWh)": res["price"]
    }).set_index("Timestamp")

@st.cache_data(ttl=3600)
def fetch_weather_data(lat=52.52, lon=13.41):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "direct_normal_irradiance", "wind_speed_100m"],
        "timezone": "Europe/Berlin",
        "forecast_days": 3
    }
    hourly = requests.get(url, params=params).json()["hourly"]
    return pd.DataFrame({
        "Timestamp": pd.to_datetime(hourly["time"]),
        "Solar Irradiance (W/m²)": hourly["direct_normal_irradiance"],
        "Wind Speed (km/h)": hourly["wind_speed_100m"]
    }).set_index("Timestamp")


with st.spinner("Fetching Live Market & Weather Inputs..."):
    df_prices = fetch_fraunhofer_prices(selected_bzn)
    df_weather = fetch_weather_data()
    df = pd.merge(df_prices, df_weather, left_index=True, right_index=True, how="inner")

# --- ML FEATURE ENGINEERING ---
features = df[["Solar Irradiance (W/m²)", "Wind Speed (km/h)"]].copy()
features["Hour"] = df.index.hour
features["DayOfWeek"] = df.index.dayofweek
target = df["Day-Ahead Price (€/MWh)"]

# --- MODEL TRAINING & PREDICTION ---
df["24h Moving Average Trend"] = df["Day-Ahead Price (€/MWh)"].rolling(window=12, min_periods=1).mean()

if "LightGBM" in selected_model:
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    model.fit(features, target)
    df["LightGBM Forecast"] = model.predict(features)

elif "XGBoost" in selected_model:
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(features, target)
    df["XGBoost Forecast"] = model.predict(features)

elif "CatBoost" in selected_model:
    model = CatBoostRegressor(iterations=100, depth=4, verbose=0, random_seed=42)
    model.fit(features, target)
    df["CatBoost Forecast"] = model.predict(features)

elif "Random Forest" in selected_model:
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(features, target)
    df["Random Forest Forecast"] = model.predict(features)

active_col = "Day-Ahead Price (€/MWh)" if "Actual" in selected_model else selected_model


# --- DASHBOARD UI ---
st.title("⚡ Short-Term Power Market Trading Simulator")
st.markdown(f"### 📍 Active Market: **{MARKET_NAMES[selected_bzn]}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Model", selected_model)
col2.metric("Active Model Price", f"{df[active_col].iloc[-1]:.2f} €/MWh")
col3.metric("Solar Irradiance", f"{df['Solar Irradiance (W/m²)'].iloc[-1]} W/m²")
col4.metric("Wind Speed (100m)", f"{df['Wind Speed (km/h)'].iloc[-1]} km/h")

st.markdown("---")

st.subheader(f"📈 Price Horizon vs. Model Predictions ({selected_bzn})")
st.line_chart(df[[active_col, "Day-Ahead Price (€/MWh)"]])

st.subheader("☀️ Renewable Generation Drivers")
st.line_chart(df[["Solar Irradiance (W/m²)", "Wind Speed (km/h)"]])

with st.expander("🔍 Explore Raw Price & Feature Matrix"):
    st.dataframe(df)
