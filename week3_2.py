import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# =====================================
# LOAD NEWS DATA
# =====================================

news = pd.read_csv("news.csv")   # Kaggle dataset
news["Date"] = pd.to_datetime(news["Date"])

# =====================================
# SENTIMENT
# =====================================

def polarity(text):
    return TextBlob(str(text)).sentiment.polarity

news["Polarity"] = news["Headline"].apply(polarity)

daily_sentiment = news.groupby("Date")["Polarity"].mean().reset_index()

# =====================================
# STOCK DATA
# =====================================

ticker = "AAPL"
stock = yf.download(ticker, start="2015-01-01")
stock.reset_index(inplace=True)

# =====================================
# MERGE
# =====================================

data = pd.merge(stock, daily_sentiment, on="Date", how="left")
data["Polarity"].fillna(0, inplace=True)

features = data[["Close", "Polarity"]]

# =====================================
# SCALE
# =====================================

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# =====================================
# CREATE SEQUENCES
# =====================================

def create_sequences(dataset, lookback=60):
    X, y = [], []
    
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i])
        y.append(dataset[i, 0])
        
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================
# LSTM MODEL
# =====================================

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 2)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =====================================
# PREDICT
# =====================================

pred = model.predict(X_test)

# inverse scaling
pred_full = np.zeros((len(pred), 2))
pred_full[:,0] = pred[:,0]

y_full = np.zeros((len(y_test), 2))
y_full[:,0] = y_test

pred_prices = scaler.inverse_transform(pred_full)[:,0]
real_prices = scaler.inverse_transform(y_full)[:,0]

rmse = np.sqrt(mean_squared_error(real_prices, pred_prices))
print("RMSE:", rmse)

# =====================================
# PLOT
# =====================================

plt.figure(figsize=(12,5))
plt.plot(real_prices, label="Actual")
plt.plot(pred_prices, label="Predicted")
plt.title("Stock Price Prediction using Sentiment + LSTM")
plt.legend()
plt.show()
