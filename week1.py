import yfinance as yf
import pandas as pd

# Download historical stock data
data = yf.download("AAPL", start="2022-01-01", end="2024-01-01")

data.head()
series = data['Close']
series.head()
series.isna().sum()
# Forward fill missing values (if any)
series = series.fillna(method='ffill')
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(series)
plt.title("AAPL Closing Price Time Series")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(series)

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
series_diff = series.diff().dropna()

adf_result_diff = adfuller(series_diff)

print("ADF Statistic:", adf_result_diff[0])
print("p-value:", adf_result_diff[1])
train_size = int(len(series) * 0.8)

train = series[:train_size]
test = series[train_size:]

len(train), len(test)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(series_diff, lags=20)
plt.show()

plot_pacf(series_diff, lags=20)
plt.show()
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())
forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(10,5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Test Data", color='black')
plt.plot(test.index, forecast, label="Forecast", color='red')
plt.legend()
plt.title("ARIMA Forecast vs Actual Closing Prices")
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
history = list(train)
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test.iloc[t])
rmse_wfv = np.sqrt(mean_squared_error(test, predictions))
rmse_wfv
residuals = model_fit.resid

plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title("Residuals of ARIMA Model")
plt.show()
residuals.plot(kind='kde')
plt.title("Residual Density")
plt.show()

