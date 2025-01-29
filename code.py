import pandas as pd
import plotly.express as px
from itertools import cycle
import streamlit as st

# Streamlit file uploader (assuming the file is already in the same directory as the script)
st.title("Stock Analysis Dashboard")
st.write("Visualize FTSE 100 stock data with an interactive date range slider.")

# Load the dataset
data = pd.read_csv('FTSE-100.csv')

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
    min_value=min_date,  # Pass datetime objects
    max_value=max_date,  # Pass datetime objects
    value=(min_date, max_date),  # Pass tuple of datetime objects
    format="YYYY-MM-DD"  # Display format for the slider values
)

# Display the min and max dates below the slider in a human-readable format
st.write(f"Min Date: {min_date.date()} | Max Date: {max_date.date()}")

# Display the selected date range in a readable format
st.write(f"Selected Date Range: {start_date.date()} to {end_date.date()}")

# Filter the data based on the selected date range
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Plotting the stock data for the selected date range
names = cycle(['Open Price', 'Close Price', 'High Price', 'Low Price'])

# Creating a line plot for open, close, high, low prices
fig = px.line(filtered_data, 
              x='Date',  # Use 'Date' column for the x-axis
              y=['Open', 'close', 'High', 'Low'],  # Plot Open, Close, High, and Low prices
              labels={'Date': 'Date', 'value': 'Stock Value'},
              title=f'Stock Analysis Chart from {start_date.date()} to {end_date.date()}')

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

