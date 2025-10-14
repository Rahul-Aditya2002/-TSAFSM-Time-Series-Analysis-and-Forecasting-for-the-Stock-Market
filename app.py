#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Suppressing TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
model = load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler_lstm_stock.pkl')


# In[ ]:


# Home Route
@app.route('/')
def home():
    return render_template('index.html')


# In[ ]:


# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['data']
        values = [float(x.strip()) for x in data.strip().split(',')]

        if len(values) != 60:
            return render_template('index.html',
                                   prediction_text="❌ Please enter exactly 60 comma-separated numbers.")

        input_data = np.array(values).reshape(-1, 1)
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, 60, 1)

        prediction = model.predict(input_reshaped)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # Plot chart
        plt.figure(figsize=(10, 4))
        plt.plot(values, label='Input Prices', marker='o')
        plt.plot([59], [predicted_price], marker='x', color='red', label='Predicted Price')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.tight_layout()

        chart_path = 'static/prediction_plot.png'
        plt.savefig(chart_path)
        plt.close()

        return render_template('index.html',
                               prediction_text=f"✅ Predicted Stock Price: ₹{predicted_price:.2f}",
                               chart_url=chart_path)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"❌ Error: {str(e)}")


# In[ ]:


# Running the app
if __name__ == '__main__':
    app.run()


# # FEW EXAMPLES
# 
# # Realistic Uptrend
# 
# 145,146,147,148,149,150,151,152,153,154,
# 155,156,158,159,160,161,163,164,165,166,
# 167,168,170,171,172,173,174,175,177,178,
# 179,180,182,183,184,185,187,188,189,190,
# 191,192,194,195,196,197,199,200,201,202,
# 203,205,206,207,208,210,211,212,213,215
# 
# # Downtrend
# 
# 220,219,218,217,216,215,214,213,212,211,
# 210,209,208,207,206,205,204,203,202,201,
# 200,199,198,197,196,195,194,193,192,191,
# 190,189,188,187,186,185,184,183,182,181,
# 180,179,178,177,176,175,174,173,172,171,
# 170,169,168,167,166,165,164,163,16,161
# 
# 
# # Volatile Stock
# 
# 100,102,97,105,95,110,92,115,88,120,
# 85,125,90,110,95,100,105,98,103,96,
# 101,100,99,97,102,95,106,93,108,91,
# 107,94,109,90,115,92,112,89,118,87,
# 119,85,120,90,110,100,105,98,99,96,
# 103,97,101,95,104,92,100,98,102,96
# 
# # Recovery after dip
# 
# 180,178,176,174,172,170,168,166,164,162,
# 160,158,156,154,152,150,155,160,165,170,
# 175,180,182,184,186,188,190,192,194,196,
# 198,200,202,204,206,208,210,212,214,216,
# 218,220,222,224,226,228,230,232,234,236,
# 238,240,242,244,246,248,250,252,254,256

# In[ ]:





# In[ ]:




