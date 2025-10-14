#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install yfinance


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = yf.download('AAPL', start='2015-01-01', end='2025-07-01')
df.head()


# # Data Cleaning and Quality Checks

# In[4]:


df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.describe()


# In[9]:


# Quick check for outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.title('Boxplot for Outlier Detection')
plt.show()


# In[10]:


# View total missing values before cleaning
print("Missing values before cleaning:\n", df.isnull().sum())

df_ffill = df.ffill()

print("\nMissing values after forward fill:\n", df_ffill.isnull().sum())


# In[11]:


# Drop duplicate rows if they exist
df_ffill = df_ffill[~df_ffill.duplicated()]

print("Duplicate rows after cleaning:", df_ffill.duplicated().sum())


# In[12]:


df_clean = df_ffill.reset_index()

df_clean.head()


# In[13]:


# Re-plot boxplot to recheck for extreme outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.title('Boxplot After Cleaning')
plt.show()


# In[14]:


volume_cap = df_clean['Volume'].quantile(0.99)
df_clean['Volume'] = np.where(df_clean['Volume'] > volume_cap, volume_cap, df_clean['Volume'])


# In[15]:


# Final summary of cleaned data
df_clean.describe()


# # Data Visualization & Exploratory Data Analysis (EDA)

# In[16]:


plt.figure(figsize=(14, 5))
plt.plot(df_clean['Date'], df_clean['Close'], color='blue')
plt.title('Apple Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[17]:


plt.figure(figsize=(14, 4))
plt.plot(df_clean['Date'], df_clean['Volume'], color='orange')
plt.title('Apple Stock Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[18]:


# Compute correlation matrix
corr_matrix = df_clean[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Price and Volume Features')
plt.show()


# In[19]:


# Calculate 30-day rolling mean
df_clean['Close_30MA'] = df_clean['Close'].rolling(window=30).mean()

# Plot original vs moving average
plt.figure(figsize=(14, 5))
plt.plot(df_clean['Date'], df_clean['Close'], label='Actual Close', color='blue')
plt.plot(df_clean['Date'], df_clean['Close_30MA'], label='30-Day MA', color='red')
plt.title('Closing Price vs 30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[20]:


# Convert to datetime (if not already)
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# Group by month
monthly_avg = df_clean.groupby(df_clean['Date'].dt.to_period("M"))['Close'].mean()

# Plot
plt.figure(figsize=(14, 5))
monthly_avg.plot()
plt.title('Monthly Average Closing Price')
plt.xlabel('Month')
plt.ylabel('Average Close Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Close'], bins=50, kde=True, color='purple')
plt.title('Distribution of Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[22]:


df_clean['Daily Return'] = df_clean['Close'].pct_change()
df_clean['Daily Return'].head()


# In[23]:


plt.figure(figsize=(14, 5))
plt.plot(df_clean['Date'], df_clean['Daily Return'], color='green')
plt.title('Daily Returns of AAPL')
plt.xlabel('Date')
plt.ylabel('Return')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Daily Return'].dropna(), bins=100, kde=True, color='darkgreen')
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


# 30-day rolling volatility
df_clean['Rolling Volatility'] = df_clean['Daily Return'].rolling(window=30).std()

plt.figure(figsize=(14, 5))
plt.plot(df_clean['Date'], df_clean['Rolling Volatility'], color='red')
plt.title('30-Day Rolling Volatility (Standard Deviation of Daily Returns)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.tight_layout()
plt.show()


# # Time Series Decomposition

# In[26]:


from statsmodels.tsa.seasonal import seasonal_decompose
df_clean.set_index('Date', inplace=True)
# Decompose closing price into trend, seasonality, and residuals
decomposition = seasonal_decompose(df_clean['Close'], model='multiplicative', period=30)  # 30-day period


# In[27]:


# Plot all components together
fig = decomposition.plot()
fig.set_size_inches(14, 8)
fig.suptitle('Time Series Decomposition of AAPL Closing Price', fontsize=16)
plt.tight_layout()
plt.show()


# # Machine Learning Forecasting Models

# ***ARIMA***

# In[29]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Use the 'Close' column as the target series
series = df_clean['Close']

series.index = pd.date_range(start=series.index[0], periods=len(series), freq='B')  # 'B' = Business Day


# In[30]:


train = series[:-100]
test = series[-100:]

model = ARIMA(train, order=(5, 1, 0))  # p=5, d=1, q=0
model_fit = model.fit()

# Forecast the next 100 steps
forecast = model_fit.forecast(steps=100)


# In[31]:


plt.figure(figsize=(14, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red')
plt.title('ARIMA Model: Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[32]:


rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"ARIMA Model RMSE: {rmse:.2f}")


# ***SARIMA***

# In[33]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[36]:


# Define SARIMA model
sarima_model = SARIMAX(train, 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 12), 
                       enforce_stationarity=False, 
                       enforce_invertibility=False)

sarima_result = sarima_model.fit()

# Forecast next 100 days
sarima_forecast = sarima_result.forecast(steps=100)


# In[37]:


plt.figure(figsize=(14, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='orange')
plt.title('SARIMA Model: Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[38]:


sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
print(f"SARIMA Model RMSE: {sarima_rmse:.2f}")


# ***The model looks underfitting till now***

# # Prophet Forecasting Model (by Meta)

# In[37]:


#!pip install prophet


# In[39]:


from prophet import Prophet


# In[40]:


# Create a new DataFrame with 'ds' and 'y' for Prophet
df_prophet = df_clean.reset_index()[['Date', 'Close']].rename(columns={
    'Date': 'ds',
    'Close': 'y'
})


# In[41]:


# Extract 'Date' and actual 'Close' column in flat format
df_temp = df_clean.reset_index().copy()

# Fix column names if multi-indexed
if isinstance(df_temp.columns, pd.MultiIndex):
    df_temp.columns = ['_'.join(col).strip() for col in df_temp.columns.values]

# Handle both flat and multi-level column cases
close_col = [col for col in df_temp.columns if 'Close' in col][0]
date_col = [col for col in df_temp.columns if 'Date' in col or 'ds' in col][0]

# Keep only Date and Close
df_temp = df_temp[[date_col, close_col]].copy()

# Rename
df_temp.columns = ['ds', 'y']

# Drop missing or non-numeric
df_temp.dropna(inplace=True)
df_temp['y'] = pd.to_numeric(df_temp['y'], errors='coerce')
df_temp.dropna(inplace=True)

# Final Prophet-ready data
df_prophet = df_temp.copy()


# In[44]:


prophet_model = Prophet()
prophet_model.fit(df_prophet)


# In[45]:


future = prophet_model.make_future_dataframe(periods=100)

# Generate forecast
forecast_prophet = prophet_model.predict(future)


# In[46]:


fig1 = prophet_model.plot(forecast_prophet)
plt.title("Prophet Forecast for AAPL Stock Price")
plt.show()


# In[47]:


start_date = test.index.min()
end_date = test.index.max()

# Slicing Prophet forecast by date range
prophet_pred = forecast_prophet.set_index('ds').loc[start_date:end_date]['yhat']


# In[48]:


plt.figure(figsize=(14, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(prophet_pred.index, prophet_pred, label='Prophet Forecast', color='green')
plt.title("Prophet Model: Actual vs Forecast")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[49]:


# Align on common dates only
common_dates = test.index.intersection(prophet_pred.index)

# Compute RMSE using aligned data
rmse = np.sqrt(mean_squared_error(test.loc[common_dates], prophet_pred.loc[common_dates]))
print(f"Prophet Model RMSE: {rmse:.2f}")


# # LSTM (Deep Learning)

# In[50]:


from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[51]:


df_lstm = df_clean.copy()

# Use only 'Close' price
data = df_lstm[['Close']].values


# In[52]:


# Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# In[53]:


train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


# In[54]:


def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

sequence_length = 60  # You can tune this later
x_train, y_train = create_sequences(train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)


# In[55]:


# Build LSTM Model
model = Sequential()

# First LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))


# In[56]:


# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[57]:


# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)


# In[58]:


# Make predictions
predictions = model.predict(x_test)

# Inverse transform to get actual price scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)


# In[59]:


plt.figure(figsize=(14, 5))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price', color='orange')
plt.title('LSTM Model: Actual vs Predicted Closing Price')
plt.xlabel('Time Steps')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[60]:


from sklearn.metrics import mean_squared_error

lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f"LSTM Model RMSE: {lstm_rmse:.2f}")


# # Model Performance Comparison Table

# In[61]:


# Model RMSE summary
model_rmse = {
    'ARIMA': 25.82,
    'SARIMA': 30.25,
    'Prophet': 20.55,
    'LSTM': 9.94
}

# Convert to DataFrame for display
rmse_df = pd.DataFrame(list(model_rmse.items()), columns=['Model', 'RMSE']).sort_values('RMSE')

# Display
print("Model Performance Comparison (Lower RMSE = Better):")
display(rmse_df)


# In[62]:


conclusion = """
📌 Conclusion & Insights:

1. LSTM model outperformed all other forecasting methods with the lowest RMSE of 10.59.
2. ARIMA and SARIMA struggled to capture the complex and volatile patterns of stock prices.
3. Prophet model did reasonably well and is good for quick prototyping.
4. LSTM is more suitable when high accuracy is desired and enough data is available.

🔍 Recommendation: Use LSTM for production-grade forecasting with continuous retraining.

"""

print(conclusion)


# In[63]:


# Save trained LSTM model
model.save("lstm_stock_model.h5")

# Save the scaler for inverse transforms later
import joblib
joblib.dump(scaler, 'scaler_lstm_stock.pkl')

print("LSTM model and scaler saved successfully.")


# In[64]:


# Export actual vs predicted prices
results_df = pd.DataFrame({
    'Actual': y_test_actual.flatten(),
    'LSTM_Predicted': predictions.flatten()
})

results_df.to_csv('lstm_stock_predictions.csv', index=False)
print("Prediction results saved to 'lstm_stock_predictions.csv'")


# # Quick Deployment

# ***Refer to C:\Users\Rahul aditya\app.py***

# ***Refer to C:\Users\Rahul aditya\flask_stock_app***

# In[ ]:




