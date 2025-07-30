import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained model
model = load_model(r"C:\Users\91730\PycharmProjects\PythonProject4\Stock Predictions Model.keras")

st.header('📈 Stock Market Predictor')

# Input
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG, MSFT)', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Fetch data
data = yf.download(stock, start, end)

st.subheader('📊 Raw Stock Data')
st.write(data)

# Train/Test Split
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Normalize and prepare test data
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test_combined = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_combined)

# Plot MA50
st.subheader('MA50 vs Close Price')
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig1)

# Plot MA50 & MA100
st.subheader('MA50 vs MA100 vs Close Price')
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(ma_100, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig2)

# Plot MA100 & MA200
st.subheader('MA100 vs MA200 vs Close Price')
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100, 'r', label='MA100')
plt.plot(ma_200, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predict
predict = model.predict(x)

# Rescale predictions and actual
scale_factor = 1 / scaler.scale_[0]
predict = predict * scale_factor
y = y * scale_factor

# Plot prediction
st.subheader('🔮 Predicted Price vs Actual Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'r', label='Actual Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
