import pandas as pd
import plotly.express as px
from itertools import cycle
import streamlit as st

# 1. Load your dataset
data = pd.read_csv('/FTSE 100.csv')

# 2. Rename the columns for consistency
data = data.rename(columns={'Price': 'close', 'Vol.': 'volume', 'Change %': 'change'})

# 3. Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# 4. Filter the data for the last year (one year prior to the latest date in the dataset)
latest_date = data['Date'].max()  # Get the most recent date in the dataset
one_year_ago = latest_date - pd.DateOffset(years=1)

# Filter the dataset to only include data from the last year
data_last_year = data[data['Date'] >= one_year_ago]

# 5. Plotting the stock data for the last year
names = cycle(['Open Price', 'Close Price', 'High Price', 'Low Price'])

# Creating a line plot for open, close, high, low prices
fig = px.line(data_last_year, 
              x='Date',  # Use 'Date' column for the x-axis
              y=['Open', 'close', 'High', 'Low'],  # Plot Open, Close, High, and Low prices
              labels={'Date': 'Date', 'value': 'Stock Value'},
              title='Stock Analysis Chart for the Last Year')

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

# Streamlit app layout
st.title("Stock Analysis Dashboard")
st.write("This dashboard visualizes stock data for the FTSE 100 over the past year.")

# Show the figure
st.plotly_chart(fig)


