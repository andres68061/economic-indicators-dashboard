import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime
from streamlit_option_menu import option_menu

# Set up Streamlit layout - must be the first Streamlit command
st.set_page_config(page_title="Economic Indicators Dashboard", layout="wide")

# Function to inject CSS from a file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS styles
local_css("styles.css")

# Hide Streamlit style (optional)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set FRED API key
fred = Fred(api_key='327e3e088495e2421efd4402e5a4c31a')  # Replace with your actual FRED API key

# Define date range
start_date = '1960-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Load data function
def load_data():
    core_pce = fred.get_series('PCEPILFE', start_date, end_date).pct_change(12) * 100
    core_pce.name = 'Core PCE Inflation Rate'

    real_gdp = fred.get_series('GDPC1', start_date, end_date)
    potential_gdp = fred.get_series('GDPPOT', start_date, end_date)
    output_gap = ((real_gdp - potential_gdp) / potential_gdp) * 100
    output_gap.name = 'Output Gap'

    unemployment_rate = fred.get_series('UNRATE', start_date, end_date)
    nairu = fred.get_series('NROU', start_date, end_date)
    unemployment_gap = unemployment_rate - nairu
    unemployment_gap.name = 'Unemployment Gap'

    bond_yield_10yr = fred.get_series('DGS10', start_date, end_date)
    bond_yield_10yr.name = '10-Year Treasury Yield'

    # Historical average rate assumption
    historical_avg_rate = pd.Series(2.0, index=core_pce.index)
    historical_avg_rate.name = 'Historical Average Rate (Assumed)'

    data = pd.concat([core_pce, output_gap, unemployment_gap, bond_yield_10yr, historical_avg_rate], axis=1)
    data = data.dropna()
    return data

# Load data
data = load_data()

# Create a horizontal menu
menu_selected = option_menu(
    menu_title=None,  # No need for a menu title
    options=["Home", "EDA", "Insights", "Prediction"],
    icons=["house", "bar-chart", "lightbulb", "activity"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#fcfcfc", "font-size": "20px"},
        "nav-link": {"font-size": "20px", "text-align": "center", "margin": "0px", "--hover-color": "#6F36A5"},
        "nav-link-selected": {"background-color": "#6F36A5"},
    }
)

# Home Section
if menu_selected == "Home":
    st.markdown("<h1 class='dashboard_title'>Welcome to the Economic Indicators Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p class='dashboard_subtitle'>
    This dashboard provides interactive visualization and analysis of key economic indicators sourced from the Federal Reserve Economic Data (FRED). Navigate through the sections to explore the data, gain insights, and make predictions based on historical trends.
    </p>
    """, unsafe_allow_html=True)

    # Display a sample of the data
    st.subheader("Sample Data")
    st.write(data.head())

    # Display a line chart of Core PCE Inflation Rate
    st.subheader("Core PCE Inflation Rate Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Core PCE Inflation Rate'], label='Core PCE Inflation Rate', color='steelblue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation Rate (%)")
    ax.legend()
    st.pyplot(fig)

# Exploratory Data Analysis Section
elif menu_selected == "EDA":
    st.markdown("<h1 class='dashboard_title'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    # Data preview with description
    st.subheader("Data Preview")
    st.markdown("<p class='kpi1_text'>Note: The 'Historical Average Rate' column is an assumed constant rate of 2% to serve as a benchmark.</p>", unsafe_allow_html=True)
    st.write(data.head())

    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Missing values
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Line plot for each component
    for col in data.columns:
        st.subheader(f"{col} Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data[col], label=col, color='skyblue' if col == 'Historical Average Rate (Assumed)' else 'steelblue')
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.legend()
        st.pyplot(fig)

    # Correlation matrix and heatmap
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    # Seasonal decomposition
    st.subheader("Seasonal Decomposition of Core PCE Inflation Rate")
    decompose_pce = seasonal_decompose(data['Core PCE Inflation Rate'].dropna(), model='additive', period=12)
    st.pyplot(decompose_pce.plot())

    st.subheader("Seasonal Decomposition of 10-Year Treasury Yield")
    decompose_yield = seasonal_decompose(data['10-Year Treasury Yield'].dropna(), model='additive', period=12)
    st.pyplot(decompose_yield.plot())

    # Histograms
    st.subheader("Distribution of Economic Indicators")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, col in enumerate(data.columns):
        data[col].hist(bins=20, ax=axes[idx], color='steelblue', edgecolor='black')
        axes[idx].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

# Insights Section
elif menu_selected == "Insights":
    st.markdown("<h1 class='dashboard_title'>Insights from EDA</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Key Observations:**

    - **Core PCE Inflation Rate** has shown significant variability over time, with notable peaks during the 1970s and early 1980s. Recent increases may be associated with economic changes following global events.

    - **Output Gap** reflects periods where actual GDP has fallen below or risen above potential GDP, aligning with economic recessions and expansions.

    - **Unemployment Gap** shows spikes during economic downturns, indicating higher unemployment rates compared to the natural rate.

    - **10-Year Treasury Yield** trends highlight the impact of monetary policies, particularly the high-interest rates in the 1980s aimed at controlling inflation.

    **Correlations:**

    - A strong positive correlation between **Core PCE Inflation Rate** and **10-Year Treasury Yield** suggests that inflation expectations influence long-term interest rates.

    - The negative correlation between **Output Gap** and **Unemployment Gap** aligns with economic theory, as higher unemployment often coincides with lower output.

    **Seasonal Patterns:**

    - Seasonal decomposition reveals cyclical trends in both inflation rates and treasury yields, which can be important for timing economic policies and investment decisions.

    **Distributions:**

    - The distribution of the **Core PCE Inflation Rate** is skewed towards lower values, indicating periods of relatively low inflation are more common.

    - The **Output Gap** and **Unemployment Gap** distributions provide insights into the typical economic conditions over the observed period.
    """)

# Prediction Section
elif menu_selected == "Prediction":
    st.markdown("<h1 class='dashboard_title'>Prediction Using Simple Linear Regression</h1>", unsafe_allow_html=True)

    # Predicting the 10-Year Treasury Yield based on Core PCE Inflation Rate
    st.subheader("Model: Predicting 10-Year Treasury Yield from Core PCE Inflation Rate")

    # Prepare the data
    data_model = data[['Core PCE Inflation Rate', '10-Year Treasury Yield']].dropna()
    X = data_model[['Core PCE Inflation Rate']]
    y = data_model['10-Year Treasury Yield']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display results
    st.subheader("Model Coefficients")
    st.write(f"Intercept: {model.intercept_:.2f}")
    st.write(f"Coefficient: {model.coef_[0]:.2f}")

    st.subheader("Model Performance on Test Set")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted 10-Year Treasury Yield")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X_test.index, y_test, label='Actual', color='steelblue')
    ax.plot(X_test.index, y_pred, label='Predicted', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('10-Year Treasury Yield (%)')
    ax.legend()
    st.pyplot(fig)

    # Allow user to input a Core PCE Inflation Rate to predict the 10-Year Treasury Yield
    st.subheader("Make a Prediction")
    input_inflation = st.number_input("Enter Core PCE Inflation Rate (%)", min_value=-5.0, max_value=15.0, value=2.0)
    predicted_yield = model.predict([[input_inflation]])
    st.write(f"Predicted 10-Year Treasury Yield: {predicted_yield[0]:.2f}%")
