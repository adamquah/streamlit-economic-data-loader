import streamlit as st
import pandas as pd

st.title("Economic Data Loader")

# Paths to your datasets (relative paths within your GitHub repo or local paths)
exchange_rate_path = "data/Exchange_Rate.csv"
interest_rate_path = "data/Interest_Rate.csv"
gdp_path = "data/GDP.csv"
unemployment_rate_path = "data/Unemployment_Rate.csv"
commodity_price_path = "data/Commodity_Price.csv"
inflation_rate_path = "data/Inflation_Rate.csv"

# Load the datasets
exchange_rate = pd.read_csv(exchange_rate_path)
interest_rate = pd.read_csv(interest_rate_path, skiprows=0)
gdp = pd.read_csv(gdp_path, skiprows=0)
unemployment_rate = pd.read_csv(unemployment_rate_path, skiprows=0)
commodity_price = pd.read_csv(commodity_price_path, skiprows=0)
inflation_rate = pd.read_csv(inflation_rate_path, skiprows=0)

# Display the head of each dataset
st.subheader("Exchange Rate Dataset")
st.write(exchange_rate.head(5))

st.subheader("Interest Rate Dataset")
st.write(interest_rate.head(5))

st.subheader("GDP Dataset")
st.write(gdp.head(5))

st.subheader("Unemployment Rate Dataset")
st.write(unemployment_rate.head(5))

st.subheader("Commodity Price Dataset")
st.write(commodity_price.head(5))

st.subheader("Inflation Rate Dataset")
st.write(inflation_rate.head(5))
