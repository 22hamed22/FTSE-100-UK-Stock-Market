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

# Convert the datetime values to timestamps (numeric format)
min_timestamp = min_date.timestamp()
max_timestamp = max_date.timestamp()

# User selects the date range via slider (based on timestamps)
start_timestamp, end_timestamp = st.slider(
    "Select Date Range",
    min_value=min_timestamp,
    max_value=max_timestamp,
    value=(min_timestamp, max_timestamp),
    format="YYYY-MM-DD"
)

# Convert the timestamps back to datetime
start_date = pd.to_datetime(start_timestamp, unit='s')
end_date = pd.to_datetime(end_timestamp, unit='s')

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
