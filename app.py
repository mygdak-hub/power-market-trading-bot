import streamlit as st
import pandas as pd
import numpy as np
import requests

# Forecasting ML Models
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

selected_models = st.sidebar.multiselect(
    "Select Forecasting Models to Compare",
    options=[
        "LightGBM Forecast",
        "XGBoost Forecast",
        "CatBoost Forecast",
        "Random Forest Forecast",
        "24h Moving Average Trend"
    ],
    default=["LightGBM Forecast"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Battery & Strategy Parameters")
battery_capacity_mwh = st.sidebar.number_input("Battery Storage Capacity (MWh)", value=10, min_value=1)
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

# --- TRAIN SELECTED MODELS ---
if "24h Moving Average Trend" in selected_models:
    df["24h Moving Average Trend"] = df["Day-Ahead Price (€/MWh)"].rolling(window=12, min_periods=1).mean()

if "LightGBM Forecast" in selected_models:
    model_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    model_lgb.fit(features, target)
    df["LightGBM Forecast"] = model_lgb.predict(features)

if "XGBoost Forecast" in selected_models:
    model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    model_xgb.fit(features, target)
    df["XGBoost Forecast"] = model_xgb.predict(features)

if "CatBoost Forecast" in selected_models:
    model_cat = CatBoostRegressor(iterations=100, depth=4, verbose=0, random_seed=42)
    model_cat.fit(features, target)
    df["CatBoost Forecast"] = model_cat.predict(features)

if "Random Forest Forecast" in selected_models:
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_rf.fit(features, target)
    df["Random Forest Forecast"] = model_rf.predict(features)


# --- DECISION SUPPORT LAYER (DSL) LOGIC ---
primary_model = selected_models[0] if selected_models else "Day-Ahead Price (€/MWh)"

def generate_decision(row, model_col, buy_t, sell_t):
    price = row[model_col]
    if price <= buy_t:
        return "🟢 CHARGE (BUY)"
    elif price >= sell_t:
        return "🔴 DISCHARGE (SELL)"
    else:
        return "⚪ HOLD (IDLE)"

df["Trading Signal"] = df.apply(generate_decision, axis=1, model_col=primary_model, buy_t=buy_threshold, sell_t=sell_threshold)

current_signal = df["Trading Signal"].iloc[-1]
current_price = df[primary_model].iloc[-1]

# Estimated daily P&L calculation based on capacity
charge_hours = df[df["Trading Signal"] == "🟢 CHARGE (BUY)"]
discharge_hours = df[df["Trading Signal"] == "🔴 DISCHARGE (SELL)"]

avg_buy_p = charge_hours[primary_model].mean() if not charge_hours.empty else 0
avg_sell_p = discharge_hours[primary_model].mean() if not discharge_hours.empty else 0
projected_spread = max(0, avg_sell_p - avg_buy_p)
estimated_daily_pnl = projected_spread * battery_capacity_mwh


# --- DASHBOARD UI ---
st.title("⚡ Short-Term Power Market Trading Simulator")
st.markdown(f"### 📍 Active Market: **{MARKET_NAMES[selected_bzn]}**")

st.markdown("---")

# 🤖 DECISION SUPPORT BANNER
st.subheader("🤖 Algorithmic Decision Support Recommendation")

banner_col1, banner_col2, banner_col3 = st.columns(3)

if "CHARGE" in current_signal:
    banner_col1.success(f"### Current Action: {current_signal}")
elif "DISCHARGE" in current_signal:
    banner_col1.error(f"### Current Action: {current_signal}")
else:
    banner_col1.info(f"### Current Action: {current_signal}")

banner_col2.metric(f"Current Signal Price ({primary_model})", f"{current_price:.2f} €/MWh")
banner_col3.metric("Projected Daily Arbitrage Spread", f"{projected_spread:.2f} €/MWh", delta=f"~€{estimated_daily_pnl:.2f} / day")

st.markdown("---")

# Visual Charting
st.subheader(f"📈 Model Prices vs. Execution Triggers")
chart_cols = ["Day-Ahead Price (€/MWh)"] + selected_models
st.line_chart(df[chart_cols])

# Execution Signals Schedule
st.subheader("📋 Next 24-Hour Automated Execution Schedule")
summary_df = df[[primary_model, "Trading Signal", "Solar Irradiance (W/m²)", "Wind Speed (km/h)"]].head(24)

st.dataframe(summary_df.style.map(
    lambda val: 'background-color: #d4edda; color: #155724;' if 'CHARGE' in str(val) 
    else ('background-color: #f8d7da; color: #721c24;' if 'DISCHARGE' in str(val) else ''),
    subset=['Trading Signal']
))
