from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

def train_model():
    stock_prices = pd.read_csv(r"C:\Users\KIIT\Desktop\AD Lab\AD_Lab3\historical_stock_prices.csv")
    stocks = pd.read_csv(r"C:\Users\KIIT\Desktop\AD Lab\AD_Lab3\historical_stocks.csv")
    merged_data = pd.merge(stock_prices, stocks, on='ticker')
    stock_data = merged_data[merged_data['ticker'] == 'AAPL']
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(by='date')
    stock_data['Target'] = stock_data['close'].shift(-1)
    stock_data = stock_data.dropna()

    features = ['close', 'volume']
    X = stock_data[features]
    y = stock_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'linear_regression_model.pkl')
    return model

if os.path.exists('linear_regression_model.pkl'):
    model = joblib.load('linear_regression_model.pkl')
else:
    model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        close = float(data['close'])
        volume = float(data['volume'])
        prediction = model.predict([[close, volume]])[0]
        return jsonify({'predicted_price': prediction})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Prediction failed. Please check your input and try again.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
