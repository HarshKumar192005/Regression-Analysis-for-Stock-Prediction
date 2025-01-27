import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

stock_prices = pd.read_csv(r"C:\Users\KIIT\Desktop\AD Lab\AD_Lab3\historical_stock_prices.csv")
stock_prices['date'] = pd.to_datetime(stock_prices['date'])

stock_data = stock_prices[stock_prices['ticker'] == 'AAPL']
stock_data = stock_data[['date', 'close']]

stock_data = stock_data.sort_values(by='date')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['close']])

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  
X, y = create_dataset(scaled_data, time_step)

X_lstm = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

X_lr = np.array([scaled_data[i:i + time_step].flatten() for i in range(len(scaled_data) - time_step)])
y_lr = scaled_data[time_step:]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)

y_pred_lr = model_lr.predict(X_test_lr)

mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
r2_lr = r2_score(y_test_lr, y_pred_lr)

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dense(units=1))  # Output layer

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

predicted_prices_lstm = model_lstm.predict(X_test)
predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)  # Inverse scaling

mse_lstm = mean_squared_error(y_test, predicted_prices_lstm)
r2_lstm = r2_score(y_test, predicted_prices_lstm)

print(f"Linear Regression Model MSE: {mse_lr}, R2: {r2_lr}")
print(f"LSTM Model MSE: {mse_lstm}, R2: {r2_lstm}")

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(y_test_lr, color='blue', label='Actual Price (LR)')
plt.plot(y_pred_lr, color='red', linestyle='dashed', label='Predicted Price (LR)')
plt.title('Linear Regression Model: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(y_test, color='blue', label='Actual Price (LSTM)')
plt.plot(predicted_prices_lstm, color='red', linestyle='dashed', label='Predicted Price (LSTM)')
plt.title('LSTM Model: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

plt.tight_layout()
plt.show()

print("Comparison of Models:")
print(f"Linear Regression - MSE: {mse_lr}, R2: {r2_lr}")
print(f"LSTM Model - MSE: {mse_lstm}, R2: {r2_lstm}")

