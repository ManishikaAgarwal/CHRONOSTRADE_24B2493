# CHRONOSTRADE_24B2493

Cronos Trade — Time Series Forecasting & Sentiment-Driven Stock Prediction (WiDS)
Overview

This repository contains the complete implementation and experimentation pipeline for Kronos Trade, a financial time-series forecasting project developed during the Women in Data Science (WiDS) program.

The project investigates:

Classical statistical forecasting using ARIMA

Deep learning approaches using Long Short-Term Memory (LSTM) networks

Sentiment analysis from financial news using TextBlob
Multimodal learning by combining price history with textual sentiment

Comparative evaluation of models under realistic forecasting setups

The work focuses on predicting stock prices (Apple Inc. – AAPL) and understanding how market psychology encoded in news headlines influences price movements.

Project Objectives

The major goals of this project are:

Build a reproducible pipeline for financial forecasting

Compare traditional and deep-learning-based models on identical datasets

Understand time-series assumptions such as stationarity and autocorrelation

Evaluate forecasting performance using robust error metrics

Explore the role of sentiment in improving predictive accuracy

Document limitations, risks, and ethical considerations in algorithmic trading systems

Repository Structure
Kronos-Trade/
│
├── data/
│   ├── raw/                 # Original downloaded datasets
│   ├── processed/           # Cleaned & merged datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_arima_model.ipynb
│   ├── 03_lstm_baseline.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_lstm_with_sentiment.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── arima_pipeline.py
│   ├── lstm_model.py
│   ├── evaluation.py
│   └── utils.py
│
├── reports/
│   └── Kronos_Trade_Final_Report.pdf
│
├── figures/
│   ├── price_vs_prediction.png
│   ├── residuals.png
│   ├── learning_curves.png
│
├── requirements.txt
├── environment.yml
└── README.md


(Adjust filenames to match your actual repo.)

Datasets Used
Stock Prices

Source: Yahoo Finance

Ticker: AAPL

Target Variable: Adjusted Close Price

Period: Multi-year daily data

News Headlines

Source: Kaggle financial news dataset

Used for extracting sentiment polarity signals

Methodology Summary
1. Data Preprocessing

Chronological train–test split (80/20)

Stationarity testing using ADF for ARIMA

First-order differencing

Min-Max scaling for neural networks

Handling missing dates in news data by assigning neutral sentiment

2. ARIMA Modeling

ACF and PACF diagnostics for (p, d, q) selection

Residual analysis

Walk-forward validation

Forecast vs actual comparisons

3. LSTM Modeling

Sliding window formulation (60-day lookback)

Two stacked LSTM layers

Adam optimizer

MSE loss

Learning-curve monitoring

4. Sentiment Analysis

TextBlob polarity scoring

Daily aggregation

Classification into Positive / Neutral / Negative categories

Integration with price data for multimodal learning

5. Evaluation

Models were compared using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Visual diagnostics included:

Price vs prediction curves

Residual plots

Training/validation loss curves

Key Findings

LSTM models outperformed ARIMA in volatile regimes

ARIMA remained a strong baseline for stable periods

Adding sentiment features reduced forecasting error

Preprocessing decisions significantly impacted final performance

Market data remains noisy and difficult to predict

Ethical Considerations & Risk Disclaimer

This project is strictly academic and experimental.

It does not constitute financial advice.

Important considerations include:

Look-ahead bias in backtesting

Overfitting risk

Absence of transaction cost modeling

Simplistic sentiment measures

Single-asset focus

Users should not deploy these models in live trading without rigorous validation.

How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/Kronos-Trade.git
cd Kronos-Trade

2. Create Environment

Using pip:

pip install -r requirements.txt


Or conda:

conda env create -f environment.yml
conda activate kronos-trade

3. Run Notebooks

Launch Jupyter:

jupyter notebook


Start with:

notebooks/01_eda.ipynb


and proceed sequentially.

Tools & Libraries

Python

Pandas, NumPy

Matplotlib, Seaborn

scikit-learn

statsmodels

TensorFlow / Keras or PyTorch

yfinance

TextBlob

Results & Report

A detailed technical discussion of all experiments, diagnostics, and findings is provided in:

reports/Kronos_Trade_Final_Report.pdf
