import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
import datetime
import yfinance as yf
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
data = yf.download(user_input, start = '2000-01-01', end='2024-05-11')
st.subheader('Data from 2000-2024')
data.head()
data
#Describing Data
st.subheader('Data Summary')
st.write(data.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=data.Close.rolling(100).mean()
ma200=data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import streamlit as st

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to forecast future stock prices
def forecast_stock(data, days=5):
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    # Example: Simple forecast by adding a constant value
    forecast_values = [data['Close'].iloc[-1] + i * 2 for i in range(1, days+1)]
    forecast_data = pd.DataFrame({'Date': forecast_dates, 'Close': forecast_values})
    return forecast_data

# Streamlit app
st.title('Stock Price Forecast')

# Sidebar inputs
symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input('End Date', datetime.now())

# Fetch historical stock data
stock_data = fetch_stock_data(symbol, start_date, end_date)

# Forecast future stock prices
forecasted_data = forecast_stock(stock_data, days=5)

# Plotting historical and forecasted data
trace1 = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data')
trace2 = go.Scatter(x=forecasted_data['Date'], y=forecasted_data['Close'], mode='lines', name='Forecasted Data')

layout = go.Layout(title=f'{symbol} Stock Price Forecast',
                   xaxis={'title': 'Date'},
                   yaxis={'title': 'Close Price'})

fig = go.Figure(data=[trace1, trace2], layout=layout)

# Display the plot in the Streamlit app
st.plotly_chart(fig)



#Splitting data into training and testing

data_training=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#Load my model
model=load_model('keras_model.h5')

#Testing Part

past_100_days=data_training.tail(100)
final_data=pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_data)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i, :]) 
    y_test.append(input_data[i,0])
    
x_test=np.array(x_test)
y_test=np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Crypto
import requests
import matplotlib.pyplot as plt

def fetch_crypto_data(coin, currency, limit=100):
    url = f"https://api.coingecko.com/api/v3/coins/{coin.lower()}/market_chart?vs_currency={currency.lower()}&days={limit}"
    response = requests.get(url)
    data = response.json()
    prices = [point[1] for point in data['prices']]
    timestamps = [point[0] for point in data['prices']]
    return prices, timestamps

def plot_crypto_prices(prices, timestamps, coin):
    fig3=plt.figure(figsize=(12, 6))
    plt.plot(timestamps, prices, label=f"{coin} Price", color='blue')
    plt.title(f"{coin} Price Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel(f"{coin} Price ({currency.upper()})")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig3)

st.title('Cryptocurrency Price Visualization')

coin = st.sidebar.text_input("Enter cryptocurrency","Bitcoin")
currency = st.sidebar.text_input("Enter currency","USD")
limit = st.sidebar.slider("Number of days", min_value=1, max_value=365, value=30)

prices, timestamps = fetch_crypto_data(coin, currency, limit)
plot_crypto_prices(prices, timestamps, coin)

