
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


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


print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

latest_data = stock_data.iloc[-1][features].values.reshape(1, -2)
future_price = model.predict(latest_data)
print("Predicted Future Price:", future_price[0])

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Prices", color='blue')
plt.plot(y_pred, label="Predicted Prices", color='red', linestyle='dashed')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
