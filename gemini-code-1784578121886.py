import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go

st.title("⚡ Short-Term Power Market Trading Simulator")

# User Inputs
st.sidebar.header("Strategy Parameters")
buy_threshold = st.sidebar.slider("Charge Trigger Price (€/MWh)", min_value=0, max_value=60, value=35)
sell_threshold = st.sidebar.slider("Discharge Trigger Price (€/MWh)", min_value=60, max_value=200, value=90)

# Generate synthetic forecast demo or connect live ENTSO-E API
st.subheader("Latest 24-Hour Price Density Forecast")

# Add your probabilistic forecast dataframe (q10, q50, q90)
# Display interactive Plotly chart with quantile bands