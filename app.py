import streamlit as st
import pandas as pd

st.title("Economic Data Loader")

# Paths to your datasets (using raw strings for Windows file paths)
exchange_rate_path = r"D:\MACHINE LEARNING\Datasets\Exchange_Rate.csv"
interest_rate_path = r"D:\MACHINE LEARNING\Datasets\Interest_Rate_selected.csv"
gdp_path = r"D:\MACHINE LEARNING\Datasets\GDP.csv"
unemployment_rate_path = r"D:\MACHINE LEARNING\Datasets\Unemployment_Rate.csv"
commodity_price_path = r"D:\MACHINE LEARNING\Datasets\Commodity_Price.csv"
inflation_rate_path = r"D:\MACHINE LEARNING\Datasets\Inflation_Rate.csv"

# Load the datasets
exchange_rate = pd.read_csv(exchange_rate_path)
interest_rate = pd.read_csv(interest_rate_path, skiprows=0)
gdp = pd.read_csv(gdp_path, skiprows=0)
unemployment_rate = pd.read_csv(unemployment_rate_path, skiprows=0)
commodity_price = pd.read_csv(commodity_price_path, skiprows=0)
inflation_rate = pd.read_csv(inflation_rate_path, skiprows=0)

# Display the datasets in the app
st.write("Exchange Rate Dataset Loaded Successfully!")
st.dataframe(exchange_rate)

st.write("Interest Rate Dataset Loaded Successfully!")
st.dataframe(interest_rate)

st.write("GDP Dataset Loaded Successfully!")
st.dataframe(gdp)

st.write("Unemployment Rate Dataset Loaded Successfully!")
st.dataframe(unemployment_rate)

st.write("Commodity Price Dataset Loaded Successfully!")
st.dataframe(commodity_price)

st.write("Inflation Rate Dataset Loaded Successfully!")
st.dataframe(inflation_rate)

# Display the head of each dataset
st.subheader("Exchange Rate Dataset (First 5 Rows)")
st.write(exchange_rate.head(5))

st.subheader("Interest Rate Dataset (First 5 Rows)")
st.write(interest_rate.head(5))

st.subheader("GDP Dataset (First 5 Rows)")
st.write(gdp.head(5))

st.subheader("Unemployment Rate Dataset (First 5 Rows)")
st.write(unemployment_rate.head(5))

st.subheader("Commodity Price Dataset (First 5 Rows)")
st.write(commodity_price.head(5))

st.subheader("Inflation Rate Dataset (First 5 Rows)")
st.write(inflation_rate.head(5))
