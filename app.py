import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Power Market Simulator", layout="wide")

st.title("⚡ Short-Term Power Market Trading Simulator")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Parameters")
buy_threshold = st.sidebar.slider("Charge Trigger Price (€/MWh)", min_value=0, max_value=60, value=35)
sell_threshold = st.sidebar.slider("Discharge Trigger Price (€/MWh)", min_value=60, max_value=200, value=90)

bzn = st.sidebar.selectbox(
    "Select Electricity Bidding Zone",
    options=["DE-LU", "AT", "FR", "NL", "BE", "DK1", "DK2", "NO2", "PL", "SE4"],
    index=0
)

# --- 1. FETCH LIVE POWER PRICES (Fraunhofer ISE Energy-Charts API - No Key) ---
@st.cache_data(ttl=1800)
def fetch_fraunhofer_prices(zone="DE-LU"):
    url = "https://api.energy-charts.info/price"
    params = {"bzn": zone}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df_prices = pd.DataFrame({
        "Timestamp": pd.to_datetime(data["unix_seconds"], unit="s"),
        "Day-Ahead Price (€/MWh)": data["price"]
    }).set_index("Timestamp")
    
    return df_prices

# --- 2. FETCH LIVE WEATHER DATA (Open-Meteo API - No Key) ---
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
    
    response = requests.get(url, params=params)
    data = response.json()
    
    hourly = data["hourly"]
    df_weather = pd.DataFrame({
        "Timestamp": pd.to_datetime(hourly["time"]),
        "Solar Irradiance (W/m²)": hourly["direct_normal_irradiance"],
        "Wind Speed (km/h)": hourly["wind_speed_100m"]
    }).set_index("Timestamp")
    
    return df_weather

# Load data
with st.spinner("Fetching live market data from Fraunhofer ISE & Open-Meteo..."):
    df_prices = fetch_fraunhofer_prices(bzn)
    df_weather = fetch_weather_data()
    
    # Merge price and weather datasets by timestamp
    df_combined = pd.merge(df_prices, df_weather, left_index=True, right_index=True, how="inner")

# --- UI DISPLAY ---
st.subheader(f"📊 Live Day-Ahead Prices & Renewable Inputs ({bzn})")

# Top KPI Metrics
col1, col2, col3 = st.columns(3)
latest_price = df_combined["Day-Ahead Price (€/MWh)"].iloc[-1]
col1.metric("Latest Day-Ahead Price", f"{latest_price:.2f} €/MWh")
col2.metric("Current Solar Irradiance", f"{df_combined['Solar Irradiance (W/m²)'].iloc[-1]} W/m²")
col3.metric("Current Wind Speed (100m)", f"{df_combined['Wind Speed (km/h)'].iloc[-1]} km/h")

# Interactive Charts
st.subheader("Electricity Spot Price Horizon")
st.line_chart(df_combined["Day-Ahead Price (€/MWh)"])

st.subheader("Renewable Energy Forecasts")
st.line_chart(df_combined[["Solar Irradiance (W/m²)", "Wind Speed (km/h)"]])

# Data Table
with st.expander("View Raw Data Table"):
    st.dataframe(df_combined)
