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
st.subheader("Dataset Previews")
st.write("### Exchange Rate Dataset")
st.write(exchange_rate.head(5))

st.write("### Interest Rate Dataset")
st.write(interest_rate.head(5))

st.write("### GDP Dataset")
st.write(gdp.head(5))

st.write("### Unemployment Rate Dataset")
st.write(unemployment_rate.head(5))

st.write("### Commodity Price Dataset")
st.write(commodity_price.head(5))

st.write("### Inflation Rate Dataset")
st.write(inflation_rate.head(5))

# Dataset Shapes
st.subheader("Dataset Shapes")
st.write("Exchange Rate Dataset Shape:", exchange_rate.shape)
st.write("Interest Rate Dataset Shape:", interest_rate.shape)
st.write("GDP Dataset Shape:", gdp.shape)
st.write("Unemployment Rate Dataset Shape:", unemployment_rate.shape)
st.write("Commodity Price Dataset Shape:", commodity_price.shape)
st.write("Inflation Rate Dataset Shape:", inflation_rate.shape)

# Dataset Descriptions
st.subheader("Dataset Descriptions")
st.write("### Exchange Rate Dataset Description")
st.write(exchange_rate.describe())

st.write("### Interest Rate Dataset Description")
st.write(interest_rate.describe())

st.write("### GDP Dataset Description")
st.write(gdp.describe())

st.write("### Unemployment Rate Dataset Description")
st.write(unemployment_rate.describe())

st.write("### Commodity Price Dataset Description")
st.write(commodity_price.describe())

st.write("### Inflation Rate Dataset Description")
st.write(inflation_rate.describe())

# Duplicates and Missing Values
st.subheader("Duplicates and Missing Values")

# Exchange Rate
st.write("### Exchange Rate Dataset")
st.write("Number of Duplicates:", exchange_rate.duplicated().sum())
st.write("Missing Values per Column:")
st.write(exchange_rate.isnull().sum())

# Interest Rate
st.write("### Interest Rate Dataset")
st.write("Number of Duplicates:", interest_rate.duplicated().sum())
st.write("Missing Values per Column:")
st.write(interest_rate.isnull().sum())

# GDP
st.write("### GDP Dataset")
st.write("Number of Duplicates:", gdp.duplicated().sum())
st.write("Missing Values per Column:")
st.write(gdp.isnull().sum())

# Unemployment Rate
st.write("### Unemployment Rate Dataset")
st.write("Number of Duplicates:", unemployment_rate.duplicated().sum())
st.write("Missing Values per Column:")
st.write(unemployment_rate.isnull().sum())

# Commodity Price
st.write("### Commodity Price Dataset")
st.write("Number of Duplicates:", commodity_price.duplicated().sum())
st.write("Missing Values per Column:")
st.write(commodity_price.isnull().sum())

# Inflation Rate
st.write("### Inflation Rate Dataset")
st.write("Number of Duplicates:", inflation_rate.duplicated().sum())
st.write("Missing Values per Column:")
st.write(inflation_rate.isnull().sum())
