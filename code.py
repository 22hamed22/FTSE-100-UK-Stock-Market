import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import cycle

# Streamlit title
st.title("FTSE 100 UK Stock Analysis and Prediction")
st.write("Visualize FTSE 100 stock data and predict future prices.")

# ---------------------------------------------------------
# Upload the dataset once
data = st.file_uploader("Upload FTSE 100 CSV Data", type="csv")
if data is not None:
    # Load the dataset
    data = pd.read_csv(data)

    # Rename columns for consistency and convenience
    data = data.rename(columns={'Price': 'close', 'Vol.': 'volume', 'Change %': 'change', 'Predicted_Close': 'Predicted_Close'})

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Remove commas in numeric columns and convert to float
    for col in ['close', 'Predicted_Close', 'Open', 'High', 'Low']:
        data[col] = data[col].replace({',': ''}, regex=True).astype(float)

    # Get the range of dates from the data
    min_date = data['Date'].min()
    max_date = data['Date'].max()

    # ---------------------------------------------------------
    # First Selection Bar for Stock Prices
    st.subheader("Stock Data (Open, Close, High, Low Prices)")
    start_date_1, end_date_1 = st.slider(
        "Date Range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
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
    # Second Selection Bar for Actual vs Predicted Close Price
    st.subheader("Actual vs Predicted Close Price")
    start_date_2, end_date_2 = st.slider(
        "Date Range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
    )

    # Filter data for second plot
    filtered_data_2 = data[(data['Date'] >= pd.to_datetime(start_date_2)) & (data['Date'] <= pd.to_datetime(end_date_2))]

    # Ensure the columns 'close' and 'Predicted_Close' are numeric
    filtered_data_2['close'] = pd.to_numeric(filtered_data_2['close'], errors='coerce')
    filtered_data_2['Predicted_Close'] = pd.to_numeric(filtered_data_2['Predicted_Close'], errors='coerce')

    # Remove NaN values
    filtered_data_2 = filtered_data_2.dropna(subset=['close', 'Predicted_Close'])

    # Second Plot: Actual vs Predicted Close Price
    

    if filtered_data_2.empty:
        st.warning("No data available for the selected date range.")
    else:
        fig2 = px.line(
            filtered_data_2,
            x='Date',
            y=['close', 'Predicted_Close'],
            labels={'Date': 'Date', 'value': 'Stock Price'},
            title=f'Actual vs Predicted Close Price from {start_date_2} to {end_date_2}'
        )

        st.plotly_chart(fig2)
