from flask import Flask, request, jsonify
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('stock_model.h5')

# Function to preprocess data
def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
    return np.array(X)
# Define a route for the root URL
@app.route('/')
def home():
    return "Welcome to the Stock Price Prediction API"
@app.route('/predict', methods=['POST'])
def predict():
    # Get the stock ticker from the POST request
    ticker = request.json['ticker']
    
    # Fetch the stock data
    data = yf.download(ticker, start='2015-01-01', end='2023-01-01')
    
    # Preprocess the data
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)
    
    # Create the dataset for prediction
    time_step = 60
    X_input = create_dataset(scaled_close_prices, time_step)
    X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
    
    # Make predictions
    predictions = model.predict(X_input)
    predictions = scaler.inverse_transform(predictions)
    
    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)

