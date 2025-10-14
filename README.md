# 📈 Time Series Analysis and Forecasting for the Stock Market

## 🧠 Overview
This project focuses on analyzing and forecasting stock market prices using a range of statistical and deep learning models.  
It demonstrates the end-to-end data science workflow — from data collection, preprocessing, visualization, and model building to deployment using a Flask web app.

The main goal is to forecast future stock prices based on historical data and identify the best-performing predictive model.

---

## 📊 Project Workflow

### 1️⃣ Data Collection
- The dataset was fetched using **Yahoo Finance (yfinance)** for stock market data.
- Features included: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### 2️⃣ Data Preprocessing
- Handled missing values and outliers.
- Converted timestamps to datetime objects.
- Checked data consistency and stationarity.
- Detected outliers using **Z-score**.

### 3️⃣ Exploratory Data Analysis (EDA)
- Visualized historical closing prices, moving averages, and volatility.
- Identified trends, seasonality, and noise components using **time series decomposition**.
- Used **Matplotlib** and **Seaborn** for line charts and distribution plots.

### 4️⃣ Forecasting Models Implemented
| Model | Description | Key Feature |
|--------|--------------|--------------|
| **ARIMA** | Autoregressive Integrated Moving Average | Captures autocorrelation and linear patterns |
| **SARIMA** | Seasonal ARIMA | Handles seasonality components |
| **Prophet** | Developed by Meta (Facebook) | Models trend and seasonality automatically |
| **LSTM** | Long Short-Term Memory Network | Deep learning model for sequential data |

---

## ⚡ Model Performance

| Model   | RMSE (Root Mean Squared Error) |
|----------|--------------------------------|
| ARIMA   | ~25.8 |
| SARIMA  | ~30.2 |
| Prophet | ~20.5 |
| LSTM    | **~9.9 (Best)** |

✅ **LSTM achieved the lowest RMSE**, proving to be the best model for stock price forecasting due to its ability to learn long-term temporal dependencies.

---

## 💻 Flask Web Application

### 🎯 Features
- Accepts **60 previous closing prices** as input.
- Predicts the **next stock price** using the trained LSTM model.
- Displays:
  - ✅ Predicted value
  - 📊 Line chart comparing recent and predicted trends

### 🧩 Files
- `app.py` → Flask backend logic
- `templates/index.html` → Frontend web interface
- `lstm_stock_model.h5` → Trained LSTM model
- `scaler_lstm_stock.pkl` → MinMaxScaler used for data normalization

---

## 🧠 Technologies Used
| Category | Tools & Libraries |
|-----------|------------------|
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Forecasting | Statsmodels (ARIMA, SARIMA), Prophet |
| Deployment | Flask |
| Data Source | Yahoo Finance (yfinance) |

---

## 🚀 How to Run the Project

### Clone the Repository
```bash
git clone <your_github_repo_link>
cd Time-Series-Stock-Market-Forecasting
