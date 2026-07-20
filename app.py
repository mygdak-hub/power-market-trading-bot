import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Power Market Simulator", layout="wide")

st.title("⚡ Short-Term Power Market Trading Simulator")

# Sidebar Strategy Parameters
st.sidebar.header("Strategy Parameters")
buy_threshold = st.sidebar.slider("Charge Trigger Price (€/MWh)", min_value=0, max_value=60, value=35)
sell_threshold = st.sidebar.slider("Discharge Trigger Price (€/MWh)", min_value=60, max_value=200, value=90)

# Function to fetch live weather inputs from Open-Meteo
@st.cache_data(ttl=3600)  # Caches data for 1 hour so it loads fast
def get_live_weather_data(lat=52.52, lon=13.41):
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
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(hourly["time"]),
        "Temperature (°C)": hourly["temperature_2m"],
        "Solar Irradiance (W/m²)": hourly["direct_normal_irradiance"],
        "Wind Speed (km/h)": hourly["wind_speed_100m"]
    }).set_index("Timestamp")
    
    return df

# Fetch and display the weather data
with st.spinner("Fetching live weather and grid forecasts..."):
    df_weather = get_live_weather_data()

st.subheader("🌐 Live Renewable Inputs (Open-Meteo API)")

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Solar Irradiance", f"{df_weather['Solar Irradiance (W/m²)'].iloc[0]} W/m²")
col2.metric("Current Wind Speed (100m)", f"{df_weather['Wind Speed (km/h)'].iloc[0]} km/h")
col3.metric("Current Temperature", f"{df_weather['Temperature (°C)'].iloc[0]} °C")

# Interactive chart
st.line_chart(df_weather[["Solar Irradiance (W/m²)", "Wind Speed (km/h)"]])

# Show raw data
with st.expander("View Raw Forecast Data"):
    st.dataframe(df_weather)
