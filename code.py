import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import cycle
import os

# Streamlit title
st.title("FTSE 100 UK Stock Analysis and Prediction")
st.write("Visualize FTSE 100 stock data and predict future prices.")

# ---------------------------------------------------------
# Automatically load the dataset from a local path
file_path = "ftse_data.csv"  # Ensure this is in the same directory as your app

if os.path.exists(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Rename columns for consistency and convenience
    data = data.rename(columns={'Price': 'close', 'Vol.': 'volume', 'Change %': 'change', 'Predicted_Close': 'Predicted_Close'})

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Remove commas in numeric columns and convert to float
    for col in ['close', 'Predicted_Close', 'Open', 'High', 'Low']:
        data[col] = data[col].replace({',': ''}, regex=True).astype(float)

    # Convert the volume column to numeric (handle 'M' and 'B' suffixes)
    def convert_volume(value):
        if isinstance(value, str):
            if 'M' in value:
                return float(value.replace('M', '').strip()) * 1e6
            elif 'B' in value:
                return float(value.replace('B', '').strip()) * 1e9
        return value

    # Apply the volume conversion function
    data['volume'] = data['volume'].apply(convert_volume)

    # Get the range of dates from the data
    min_date = data['Date'].min()
    max_date = data['Date'].max()

    # ---------------------------------------------------------
    # First Selection Bar for Stock Prices
    st.subheader("Stock Data (Open, Close, High, Low Prices)")
    start_date_1, end_date_1 = st.slider(
        "Date Range for Stock Prices",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD",
        key="slider1"  # Unique key for the first slider
    )

    # Filter data for first plot
    filtered_data_1 = data[(data['Date'] >= pd.to_datetime(start_date_1)) & (data['Date'] <= pd.to_datetime(end_date_1))]

    # First Plot: Stock Data (Open, Close, High, Low Prices)
    names = cycle(['Open Price', 'Close Price', 'High Price', 'Low Price'])

    fig1 = px.line(
        filtered_data_1,
        x='Date',
        y=['Open', 'close', 'High', 'Low'],
        labels={'Date': 'Date', 'value': 'Stock Value'},
        title=f'Stock Analysis Chart from {start_date_1} to {end_date_1}'
    )

    fig1.update_layout(font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig1.for_each_trace(lambda t: t.update(name=next(names)))
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(showgrid=False)

    st.plotly_chart(fig1)

    # ---------------------------------------------------------
    # Second Selection Bar for Volume Data
    st.subheader("Volume:")
    start_date_2, end_date_2 = st.slider(
        "Date Range for Volume Data",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD",
        key="slider2"  # Unique key for the second slider
    )

    # Filter data for the volume plot
    filtered_data_2 = data[(data['Date'] >= pd.to_datetime(start_date_2)) & (data['Date'] <= pd.to_datetime(end_date_2))]

    # Volume Plot: Plot the volume data
    fig2 = px.line(
        filtered_data_2,
        x='Date',
        y='volume',
        labels={'Date': 'Date', 'volume': 'Volume'},
        title=f'Volume Data from {start_date_2} to {end_date_2}'
    )

    fig2.update_layout(font_size=15, font_color='black', showlegend=False)
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)

    st.plotly_chart(fig2)

    # ---------------------------------------------------------
    # Third Selection Bar for Actual vs Predicted Close Price
    st.subheader("Actual vs Predicted Close Price")
    start_date_3, end_date_3 = st.slider(
        "Date Range for Actual vs Predicted Close Price",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD",
        key="slider3"  # Unique key for the third slider
    )

    # Filter data for second plot
    filtered_data_3 = data[(data['Date'] >= pd.to_datetime(start_date_3)) & (data['Date'] <= pd.to_datetime(end_date_3))]

    # Ensure the columns 'close' and 'Predicted_Close' are numeric
    filtered_data_3['close'] = pd.to_numeric(filtered_data_3['close'], errors='coerce')
    filtered_data_3['Predicted_Close'] = pd.to_numeric(filtered_data_3['Predicted_Close'], errors='coerce')

    # Remove NaN values
    filtered_data_3 = filtered_data_3.dropna(subset=['close', 'Predicted_Close'])

    # Actual vs Predicted Close Price Plot
    if filtered_data_3.empty:
        st.warning("No data available for the selected date range.")
    else:
        fig3 = px.line(
            filtered_data_3,
            x='Date',
            y=['close', 'Predicted_Close'],
            labels={'Date': 'Date', 'value': 'Stock Price'},
            title=f'Actual vs Predicted Close Price from {start_date_3} to {end_date_3}'
        )

        st.plotly_chart(fig3)

    # ---------------------------------------------------------
    # Metrics Section (after the charts)
    st.subheader("AI Model Evaluation Metrics")
    
    # Model Evaluation Metrics
    metrics = ["MAE", "MSE", "RMSE", "MAPE", "R² Score"]
    values = [70.40, 7307.78, 85.49, 1.11, 0.9920]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig = go.Figure([go.Scatter(x=[0.5], y=[1 - i * 0.18], text=[f"<b>{m}:</b> {v:.2f}"], 
                                mode="text", textfont=dict(size=28, color=c)) 
                     for i, (m, v, c) in enumerate(zip(metrics, values, colors))])

    fig.update_layout(title="<b>AI Model Evaluation Metrics</b>", title_x=0.5, height=500, 
                      xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white")

    st.plotly_chart(fig)

else:
    st.error(f"File {file_path} not found.")
