import pandas as pd
import plotly.express as px
from itertools import cycle
import streamlit as st

# Streamlit file uploader
st.title("Stock Analysis Dashboard")
st.write("Upload a CSV file to visualize stock data for the FTSE 100.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Rename the columns for consistency
    data = data.rename(columns={'Price': 'close', 'Vol.': 'volume', 'Change %': 'change'})
    
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Get the range of dates from the data for the date selection
    min_date = data['Date'].min()
    max_date = data['Date'].max()
    
    # User selects the start and end dates using a slider
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter the data based on the selected date range
    start_date, end_date = date_range
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
