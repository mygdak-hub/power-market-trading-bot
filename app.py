import streamlit as st
import pandas as pd
import numpy as np
import requests

# Forecasting ML Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor

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
st.sidebar.header("⚙️ Market & Product Settings")

selected_bzn = st.sidebar.selectbox(
    "Target Power Market",
    options=list(MARKET_NAMES.keys()),
    format_func=lambda x: MARKET_NAMES[x],
    index=0
)

# 1. TRADING HORIZON SELECTION
market_horizon = st.sidebar.radio(
    "Select Market Product Horizon",
    options=[
        "Day-Ahead Spot Market (12:00 Auction)",
        "Intraday Continuous / Real-Time",
        "Next-Day Forward Forecast (D+1 / D+2)"
    ],
    index=0
)

# 2. MODEL SELECTION
selected_models = st.sidebar.multiselect(
    "Select Forecasting Models to Compare",
    options=[
        "LEAR (Lasso AutoRegressive)",
        "Deep Neural Net (DNN)",
        "Stacked EPF Ensemble",
        "LightGBM Forecast",
        "CatBoost Forecast",
        "24h Moving Average Trend"
    ],
    default=["LEAR (Lasso AutoRegressive)", "Stacked EPF Ensemble"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Battery Storage Parameters")
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


with st.spinner("Fetching Market Data..."):
    df_prices = fetch_fraunhofer_prices(selected_bzn)
    df_weather = fetch_weather_data()
    df = pd.merge(df_prices, df_weather, left_index=True, right_index=True, how="inner")

# --- DERIVE INTRADAY & FORWARD MARKET DATA ---
# Synthetic Intraday spread volatility model based on real solar/wind ramp rates
np.random.seed(42)
ramp_factor = (df["Solar Irradiance (W/m²)"].diff().fillna(0) / 100)
df["Intraday Continuous (€/MWh)"] = df["Day-Ahead Price (€/MWh)"] + (ramp_factor * 5) + np.random.normal(0, 3, len(df))

# Forward D+1 Shifted Curve
df["Next-Day Forward Curve (€/MWh)"] = df["Day-Ahead Price (€/MWh)"].shift(-24).bfill()

# --- FEATURE ENGINEERING ---
df_feat = df.copy()
df_feat["Price_Lag24"] = df_feat["Day-Ahead Price (€/MWh)"].shift(24).bfill().ffill()
df_feat["Price_Lag48"] = df_feat["Day-Ahead Price (€/MWh)"].shift(48).bfill().ffill()
df_feat["Hour"] = df_feat.index.hour
df_feat["DayOfWeek"] = df_feat.index.dayofweek

feature_cols = ["Solar Irradiance (W/m²)", "Wind Speed (km/h)", "Price_Lag24", "Price_Lag48", "Hour", "DayOfWeek"]
X = df_feat[feature_cols].fillna(0)

# Set target based on selected horizon
if "Intraday" in market_horizon:
    y = df_feat["Intraday Continuous (€/MWh)"].fillna(0)
    base_price_col = "Intraday Continuous (€/MWh)"
elif "Next-Day" in market_horizon:
    y = df_feat["Next-Day Forward Curve (€/MWh)"].fillna(0)
    base_price_col = "Next-Day Forward Curve (€/MWh)"
else:
    y = df_feat["Day-Ahead Price (€/MWh)"].fillna(0)
    base_price_col = "Day-Ahead Price (€/MWh)"


# --- TRAIN ML MODELS ---
if "24h Moving Average Trend" in selected_models:
    df["24h Moving Average Trend"] = df[base_price_col].rolling(window=24, min_periods=1).mean()

if "LEAR (Lasso AutoRegressive)" in selected_models:
    model_lear = LassoCV(cv=3, random_state=42)
    model_lear.fit(X, y)
    df["LEAR (Lasso AutoRegressive)"] = model_lear.predict(X)

if "Deep Neural Net (DNN)" in selected_models:
    model_dnn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model_dnn.fit(X, y)
    df["Deep Neural Net (DNN)"] = model_dnn.predict(X)

if "LightGBM Forecast" in selected_models:
    model_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    model_lgb.fit(X, y)
    df["LightGBM Forecast"] = model_lgb.predict(X)

if "CatBoost Forecast" in selected_models:
    model_cat = CatBoostRegressor(iterations=100, depth=4, verbose=0, random_seed=42)
    model_cat.fit(X, y)
    df["CatBoost Forecast"] = model_cat.predict(X)

if "Stacked EPF Ensemble" in selected_models:
    estimators = [
        ('lear', LassoCV(cv=3, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1, random_state=42)),
        ('cat', CatBoostRegressor(iterations=50, depth=3, verbose=0, random_seed=42))
    ]
    stack_model = StackingRegressor(estimators=estimators, final_estimator=LassoCV(cv=3, random_state=42))
    stack_model.fit(X, y)
    df["Stacked EPF Ensemble"] = stack_model.predict(X)


# --- DECISION SUPPORT LAYER LOGIC ---
primary_model = selected_models[0] if selected_models else base_price_col

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

charge_hours = df[df["Trading Signal"] == "🟢 CHARGE (BUY)"]
discharge_hours = df[df["Trading Signal"] == "🔴 DISCHARGE (SELL)"]

avg_buy_p = charge_hours[primary_model].mean() if not charge_hours.empty else 0
avg_sell_p = discharge_hours[primary_model].mean() if not discharge_hours.empty else 0
projected_spread = max(0, avg_sell_p - avg_buy_p)
estimated_daily_pnl = projected_spread * battery_capacity_mwh


# --- DASHBOARD UI ---
st.title("⚡ Short-Term Power Market Trading Simulator")
st.markdown(f"### 📍 Active Market: **{MARKET_NAMES[selected_bzn]}** | Product: **{market_horizon}**")

st.markdown("---")

# Decision Support Banner
st.subheader("🤖 Algorithmic Decision Support Recommendation")

banner_col1, banner_col2, banner_col3 = st.columns(3)

if "CHARGE" in current_signal:
    banner_col1.success(f"### Current Action: {current_signal}")
elif "DISCHARGE" in current_signal:
    banner_col1.error(f"### Current Action: {current_signal}")
else:
    banner_col1.info(f"### Current Action: {current_signal}")

banner_col2.metric(f"Signal Price ({primary_model})", f"{current_price:.2f} €/MWh")
banner_col3.metric("Projected Arbitrage Spread", f"{projected_spread:.2f} €/MWh", delta=f"~€{estimated_daily_pnl:.2f} / day")

st.markdown("---")

# Charting
st.subheader(f"📈 Price Curve & Models ({market_horizon})")
chart_cols = [base_price_col] + selected_models
st.line_chart(df[chart_cols])

# Execution Table
st.subheader("📋 Next 24-Hour Execution Schedule")
summary_df = df[[primary_model, "Trading Signal", "Solar Irradiance (W/m²)", "Wind Speed (km/h)"]].head(24)

st.dataframe(summary_df.style.map(
    lambda val: 'background-color: #d4edda; color: #155724;' if 'CHARGE' in str(val) 
    else ('background-color: #f8d7da; color: #721c24;' if 'DISCHARGE' in str(val) else ''),
    subset=['Trading Signal']
))
