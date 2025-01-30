import pandas as pd
import plotly.express as px
from itertools import cycle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Define the LSTM model using PyTorch
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
            self.fc = nn.Linear(hidden_layer_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            predictions = self.fc(lstm_out[:, -1])
            return predictions

    # Instantiate and train the model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for 10 epochs
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Predicting the stock prices
    model.eval()
    with torch.no_grad():
        predicted_stock_price = model(X_train).numpy()

    # Inverse transform the predictions
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    real_stock_price = scaler.inverse_transform(y_train.reshape(-1, 1))

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
