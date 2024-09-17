import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

# Add EDA header
st.header("DATA CLEANING)")

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




# Add EDA header
st.header("Interest Rate")

# Load the dataset (replace with your actual data path or logic to load the data)
interest_rate_path = "data/Interest_Rate.csv"
interest_rate = pd.read_csv(interest_rate_path)

# Display the dataset's missing values
st.subheader("Missing Values in Interest Rate Dataset")
missing_values = interest_rate.isnull().sum()
st.write(missing_values)

# Clean the dataset by dropping columns where all values are NaN and filling NaN with 0
st.subheader("Cleaned Interest Rate Dataset")
interest_rate_cleaned = interest_rate.dropna(axis=1, how='all')
interest_rate_cleaned.fillna(0, inplace=True)

# Display the first 50 rows of the cleaned dataset
st.write("First 50 rows of the cleaned Interest Rate Dataset")
st.write(interest_rate_cleaned.head(50))


import streamlit as st

# Define the actual columns and new interest rate lists
actual_columns = ['Year', 'Aruba', 'Afghanistan', 'Angola', 'Albania', 'Argentina', 'Armenia', 
                   'Antigua and Barbuda', 'Australia', 'Azerbaijan', 'Burundi', 'Benin', 
                   'Burkina Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas, The', 
                   'Bosnia and Herzegovina', 'Belarus', 'Belize', 'Bolivia', 'Brazil', 
                   'Barbados', 'Brunei Darussalam', 'Bhutan', 'Botswana', 'Canada', 
                   'Switzerland', 'Chile', 'China', "Cote d'Ivoire", 'Congo, Dem. Rep.', 
                   'Colombia', 'Comoros', 'Cabo Verde', 'Costa Rica', 'Czechia', 'Dominica', 
                   'Dominican Republic', 'Algeria', 'Egypt, Arab Rep.', 'Ethiopia', 'Fiji', 
                   'Micronesia, Fed. Sts.', 'United Kingdom', 'Georgia', 'Guinea', 'Gambia, The', 
                   'Guinea-Bissau', 'Grenada', 'Guatemala', 'Guyana', 'Hong Kong SAR, China', 
                   'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'India', 
                   'Iran, Islamic Rep.', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 
                   'Jordan', 'Japan', 'Kenya', 'Kyrgyz Republic', 'St. Kitts and Nevis', 
                   'Korea, Rep.', 'Kuwait', 'Lao PDR', 'Lebanon', 'Liberia', 'Libya', 
                   'St. Lucia', 'Sri Lanka', 'Lesotho', 'Macao SAR, China', 'Moldova', 
                   'Madagascar', 'Maldives', 'Mexico', 'North Macedonia', 'Mali', 'Malta', 
                   'Myanmar', 'Montenegro', 'Mongolia', 'Mozambique', 'Mauritania', 
                   'Mauritius', 'Malawi', 'Malaysia', 'Namibia', 'Niger', 'Nigeria', 
                   'Nicaragua', 'Netherlands', 'Norway', 'New Zealand', 'Oman', 'Pakistan', 
                   'Panama', 'Peru', 'Philippines', 'Papua New Guinea', 'Paraguay', 
                   'West Bank and Gaza', 'Qatar', 'Romania', 'Russian Federation', 
                   'Rwanda', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone', 
                   'San Marino', 'Somalia', 'Serbia', 'South Sudan', 'Sao Tome and Principe', 
                   'Suriname', 'Sweden', 'Eswatini', 'Seychelles', 'Togo', 'Thailand', 
                   'Tajikistan', 'Timor-Leste', 'Tonga', 'Trinidad and Tobago', 'Tanzania', 
                   'Uganda', 'Ukraine', 'Uruguay', 'United States', 'Uzbekistan', 
                   'St. Vincent and the Grenadines', 'Venezuela, RB', 'Viet Nam', 
                   'Vanuatu', 'Samoa', 'Kosovo', 'Yemen, Rep.', 'South Africa', 'Zambia', 'Zimbabwe']

new_interest_rate = [
    'Year', 'Australia', 'United Arab Emirates', 'Bangladesh', 'Brazil', 'Canada',
    'Euro area', 'Switzerland', 'China', 'United Kingdom', 'Hong Kong SAR, China',
    'Indonesia', 'India', 'Japan', 'Korea, Rep.', 'Mexico', 'Malaysia',
    'Philippines', 'Russian Federation', 'Saudi Arabia', 'Singapore',
    'Thailand', 'Turkiye', 'United States'
]

# Calculate missing columns
missing_columns = [col for col in actual_columns if col not in new_interest_rate]

# Streamlit app
st.title('Missing Columns Finder')

st.write("**Actual Columns List:**")
st.write(actual_columns)

st.write("**New Interest Rate Columns List:**")
st.write(new_interest_rate)

st.write("**Missing Columns:**")
st.write(missing_columns)


# List of new interest rate columns
new_interest_rate = [
    'Year', 'Australia', 'Bangladesh', 'Brazil', 'Canada', 'Switzerland',
    'China', 'United Kingdom', 'Hong Kong SAR, China', 'Indonesia', 'India',
    'Japan', 'Korea, Rep.', 'Mexico', 'Malaysia', 'Philippines', 'Russian Federation',
    'Singapore', 'Thailand', 'United States'
]

# Streamlit app
st.title('Interest Rate Data Cleaner')

st.write("**Original Columns:**")
st.write(interest_rate_cleaned.columns.tolist())

# Clean column names
interest_rate_cleaned.columns = interest_rate_cleaned.columns.str.strip()

# Show cleaned columns
st.write("**Cleaned Columns:**")
st.write(interest_rate_cleaned.columns.tolist())

# Select relevant columns
interest_rate_selected = interest_rate_cleaned[new_interest_rate]

# Save the cleaned data
if st.button('Save to CSV'):
    file_path = "/content/drive/MyDrive/Classroom/BMCS2114MachineLearning/Assignment/DataSets/Interest_Rate_selected.csv"
    interest_rate_selected.to_csv(file_path, index=False)
    st.success(f"Data saved to {file_path}")

# Show the cleaned and selected data
st.write("**Selected Data (First 20 rows):**")
st.write(interest_rate_selected.head(20))


# Title of the Streamlit app
st.title("Interactive 3D Outlier Detection in Interest Rates")

# Load the dataset (replace with your actual path or integrate with data loading section)
interest_rate_path = "data/Interest_Rate.csv"
interest_rate_selected = pd.read_csv(interest_rate_path, index_col=0)

# Compute z-scores
z_scores = interest_rate_selected.apply(zscore, axis=0)

# Threshold for outliers
threshold = 3

# Count outliers
outliers = (np.abs(z_scores) > threshold).sum(axis=0)

# Extract years and countries for plotting
years = interest_rate_selected.index
countries = interest_rate_selected.columns

# Initialize Z matrix for outliers
Z = np.zeros((len(countries), len(years)))

for i, country in enumerate(countries):
    Z[i, :] = (np.abs(z_scores[country]) > threshold).astype(int).values

# Prepare data for 3D plot
X, Y = np.meshgrid(range(len(years)), range(len(countries)))
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Create 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=X_flat,
    y=Y_flat,
    z=Z_flat,
    mode='markers',
    marker=dict(
        size=8,
        color=Z_flat,
        colorscale='Viridis',
        colorbar=dict(title='Count of Outliers')
    )
)])

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Year',
        yaxis_title='Country',
        zaxis_title='Count of Outliers',
        xaxis=dict(tickvals=list(range(0, len(years), 5)), ticktext=years[::5]),
        yaxis=dict(tickvals=list(range(0, len(countries), 3)), ticktext=countries[::3])
    ),
    title='Interactive 3D Chart of Outliers in Interest Rates',
    autosize=True
)

# Display the 3D plot in the Streamlit app
st.plotly_chart(fig)


st.title("Interest Rate Trends Over Time")

# Assuming the dataset is already loaded
# Load the interest_rate_selected dataset (adjust the path as necessary)
interest_rate_path = "data/Interest_Rate.csv"
interest_rate_selected = pd.read_csv(interest_rate_path)

# If 'Year' is a column in the dataset, set it as the index
if 'Year' in interest_rate_selected.columns:
    interest_rate_selected.set_index('Year', inplace=True)

# Calculate average, min, and max interest rates across all countries for each year
average_rates = interest_rate_selected.mean(axis=1)
min_rates = interest_rate_selected.min(axis=1)
max_rates = interest_rate_selected.max(axis=1)

# Display calculated statistics
st.subheader("Interest Rate Statistics")
st.write("Average Rates:")
st.write(average_rates.head())

st.write("Minimum Rates:")
st.write(min_rates.head())

st.write("Maximum Rates:")
st.write(max_rates.head())

# Plot the interest rate trends
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(average_rates.index, average_rates, label='Average Interest Rate', color='blue', linestyle='-', marker='o')
ax.plot(min_rates.index, min_rates, label='Minimum Interest Rate', color='red', linestyle='--', marker='x')
ax.plot(max_rates.index, max_rates, label='Maximum Interest Rate', color='green', linestyle='-.', marker='s')

ax.fill_between(average_rates.index, min_rates, max_rates, color='grey', alpha=0.2)

ax.set_title("Aggregated Interest Rate Trends Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Interest Rate (%)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)
