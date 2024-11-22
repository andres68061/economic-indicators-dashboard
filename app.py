import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
from streamlit_option_menu import option_menu
import numpy as np

# Import TensorFlow for GRU model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

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
    # Core PCE Inflation Rate (monthly)
    core_pce = fred.get_series('PCEPILFE', start_date, end_date)
    core_pce = core_pce.resample('M').last()
    core_pce = core_pce.pct_change(12) * 100
    core_pce.name = 'Core PCE Inflation Rate'

    # Real GDP and Potential GDP (quarterly, forward-filled to monthly)
    real_gdp = fred.get_series('GDPC1', start_date, end_date)
    real_gdp = real_gdp.resample('M').ffill()
    potential_gdp = fred.get_series('GDPPOT', start_date, end_date)
    potential_gdp = potential_gdp.resample('M').ffill()
    output_gap = ((real_gdp - potential_gdp) / potential_gdp) * 100
    output_gap.name = 'Output Gap'

    # Unemployment Rate and NAIRU (monthly and quarterly)
    unemployment_rate = fred.get_series('UNRATE', start_date, end_date)
    nairu = fred.get_series('NROU', start_date, end_date)
    nairu = nairu.resample('M').ffill()
    unemployment_gap = unemployment_rate - nairu
    unemployment_gap.name = 'Unemployment Gap'

    # 10-Year Treasury Yield (daily, converted to monthly average)
    bond_yield_10yr = fred.get_series('DGS10', start_date, end_date)
    bond_yield_10yr = bond_yield_10yr.resample('M').mean()
    bond_yield_10yr.name = '10-Year Treasury Yield'

    # Federal Funds Rate (monthly)
    federal_funds_rate = fred.get_series('FEDFUNDS', start_date, end_date)
    federal_funds_rate.name = 'Federal Funds Rate'

    # Combine data
    data = pd.concat([core_pce, output_gap, unemployment_gap, bond_yield_10yr, federal_funds_rate], axis=1)
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
        ax.plot(data.index, data[col], label=col, color='steelblue')
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
    st.markdown("<h1 class='dashboard_title'>Prediction Using GRU Neural Network</h1>", unsafe_allow_html=True)

    st.subheader("Model: Predicting Federal Funds Rate Using a GRU Neural Network")

    # Prepare the data
    data_model = data[['Core PCE Inflation Rate', 'Output Gap', 'Unemployment Gap', '10-Year Treasury Yield', 'Federal Funds Rate']]

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_model)

    # Create sequences
    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, :-1])  # All features except target
            y.append(data[i, -1])  # Target variable (Federal Funds Rate)
        return np.array(X), np.array(y)

    seq_length = 12  # Use past 12 months to predict the next month
    X, y = create_sequences(scaled_data, seq_length)

    # Split the data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build the GRU model
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(GRU(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    st.subheader("Model Performance")
    st.write(f"Train Loss (MSE): {train_loss:.6f}")
    st.write(f"Test Loss (MSE): {test_loss:.6f}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform the predictions
    y_test_full = np.hstack((X_test[:, -1, :-1], y_test.reshape(-1, 1)))
    y_pred_full = np.hstack((X_test[:, -1, :-1], y_pred))
    y_test_actual = scaler.inverse_transform(y_test_full)[:, -1]
    y_pred_actual = scaler.inverse_transform(y_pred_full)[:, -1]

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Federal Funds Rate")
    fig, ax = plt.subplots(figsize=(10, 6))
    test_dates = data.index[seq_length + split:]
    ax.plot(test_dates, y_test_actual, label='Actual', color='steelblue')
    ax.plot(test_dates, y_pred_actual, label='Predicted', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Federal Funds Rate (%)')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**

    - The GRU model captures the general trend of the Federal Funds Rate over time.
    - Some discrepancies between actual and predicted values may be due to external factors not included in the model.
    """)

    # Allow user to input values to predict the Federal Funds Rate
    st.subheader("Make a Prediction")
    st.write("Enter values for the following indicators to predict the Federal Funds Rate:")

    input_inflation = st.number_input("Core PCE Inflation Rate (%)", min_value=-5.0, max_value=15.0, value=2.0)
    input_output_gap = st.number_input("Output Gap (%)", min_value=-10.0, max_value=10.0, value=0.0)
    input_unemployment_gap = st.number_input("Unemployment Gap (%)", min_value=-5.0, max_value=5.0, value=0.0)
    input_bond_yield = st.number_input("10-Year Treasury Yield (%)", min_value=0.0, max_value=15.0, value=2.0)

    # Prepare the input data
    input_data = np.array([[input_inflation, input_output_gap, input_unemployment_gap, input_bond_yield]])
    input_data_scaled = scaler.transform(np.hstack((input_data, [[0]])))[:, :-1]  # Exclude the dummy target

    # Use the last sequence from the test set and append the new input
    input_sequence = X_test[-1, 1:, :]  # Remove the oldest entry
    input_sequence = np.vstack([input_sequence, input_data_scaled])
    input_sequence = input_sequence.reshape(1, seq_length, X_train.shape[2])

    # Predict
    predicted_rate_scaled = model.predict(input_sequence)
    predicted_full = np.hstack((input_data_scaled, predicted_rate_scaled))
    predicted_actual = scaler.inverse_transform(predicted_full)
    predicted_funds_rate = predicted_actual[0, -1]

    st.write(f"**Predicted Federal Funds Rate:** {predicted_funds_rate:.2f}%")

    st.markdown("""
    **Note:** This prediction is based on the GRU model trained on historical data. Actual Federal Funds Rate decisions are influenced by a wide range of economic factors and policy considerations.
    """)
