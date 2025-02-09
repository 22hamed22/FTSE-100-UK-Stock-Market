import pandas as pd
import plotly.express as px
from itertools import cycle
import streamlit as st
import numpy as np


# Check if PyTorch is installed
try:
    import torch
    print("PyTorch is installed.")
except ImportError:
    print("PyTorch is not installed.")
    

# Streamlit file uploader
st.title("FTSE 100 UK Stock Analysis and Prediction")
st.write("Visualize FTSE 100 stock data and predict future prices.")

# Upload the dataset via Streamlit
data = st.file_uploader("Upload FTSE 100 CSV", type="csv")
if data is not None:
    # Load the dataset
    data = pd.read_csv(data)

    # Rename the columns for consistency
    data = data.rename(columns={'Price': 'close', 'Vol.': 'volume', 'Change %': 'change'})

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Get the range of dates from the data for the slider
    min_date = data['Date'].min()
    max_date = data['Date'].max()

    # User selects the date range via slider
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date.date(),  # Use human-readable min date
        max_value=max_date.date(),  # Use human-readable max date
        value=(min_date.date(), max_date.date()),  # Use human-readable date range as default
        format="YYYY-MM-DD"  # Display format for the slider values
    )

    # Filter the data based on the selected date range
    filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

    # Plotting the stock data for the selected date range
    names = cycle(['Open Price', 'Close Price', 'High Price', 'Low Price'])

    # Creating a line plot for open, close, high, low prices
    fig = px.line(filtered_data, 
                  x='Date',  # Use 'Date' column for the x-axis
                  y=['Open', 'close', 'High', 'Low'],  # Plot Open, Close, High, and Low prices
                  labels={'Date': 'Date', 'value': 'Stock Value'},
                  title=f'Stock Analysis Chart from {start_date} to {end_date}')

    # Updating the layout and legend
    fig.update_layout(
        font_size=15, 
        font_color='black', 
        legend_title_text='Stock Parameters'
    )

    # Setting custom trace names from the cycle
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    # Removing grid lines for a cleaner look
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Show the figure
    st.plotly_chart(fig)

