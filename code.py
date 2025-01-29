import pandas as pd
import plotly.express as px
from itertools import cycle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

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

    # Preparing the data for the model
    # Use 'Close' price for prediction
    price_data = filtered_data['close'].values
    price_data = price_data.reshape(-1, 1)

    # Scaling the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_data)

    # Create sequences for training (using 60 previous days to predict the next day's price)
    sequence_length = 60
    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshaping X for LSTM input (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define and train the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer with one unit (next day's price)

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predicting the stock prices
    predicted_stock_price = model.predict(X_test)
    
    # Inverse transform the predictions
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plotting the results (real vs predicted)
    predicted_df = pd.DataFrame({'Date': filtered_data['Date'].iloc[-len(predicted_stock_price):], 
                                 'Predicted Close': predicted_stock_price.flatten()})
    
    real_df = pd.DataFrame({'Date': filtered_data['Date'].iloc[-len(real_stock_price):], 
                            'Real Close': real_stock_price.flatten()})

    # Merge the dataframes for plotting
    result_df = pd.merge(real_df, predicted_df, on='Date')

    # Plot the real vs predicted stock prices
    fig2 = px.line(result_df, 
                   x='Date', 
                   y=['Real Close', 'Predicted Close'], 
                   labels={'Date': 'Date', 'value': 'Stock Value'},
                   title=f'Real vs Predicted Stock Prices')

    # Show the second figure
    st.plotly_chart(fig2)
