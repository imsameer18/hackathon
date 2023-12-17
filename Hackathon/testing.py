import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def train_lstm_model(data, sequence_length=30, epochs=10, batch_size=32):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, sequence_length)

    split_index = int(0.8 * len(data))
    X_train = X[:split_index]
    y_train = y[:split_index]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model, scaler

def predict_stock_price(model, scaler, last_30_days):
    last_30_days_scaled = scaler.transform(last_30_days)
    last_30_days_scaled = last_30_days_scaled.reshape(1, last_30_days_scaled.shape[0], 1)

    predicted_price_after_30_days_scaled = model.predict(last_30_days_scaled)
    predicted_price_after_30_days = scaler.inverse_transform(predicted_price_after_30_days_scaled)

    return predicted_price_after_30_days[0][0]

def fetch_stock_predictions_below_price(stock_symbol, max_price):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = stock_data['Close'].values.reshape(-1, 1)

    if data[-1][0] > max_price:
        return None

    sequence_length = 30
    model, scaler = train_lstm_model(data)

    last_30_days = data[-sequence_length:]
    predicted_price = predict_stock_price(model, scaler, last_30_days)

    result = {
        'stock_symbol': stock_symbol,
        'last_30_days_price': last_30_days[-1][0],
        'predicted_price_after_30_days': predicted_price,
        'max_price': max_price,
    }

    return result

# max_price = 1500.0

# results_list = []

# for stock_symbol in stocks:
#     result = fetch_stock_predictions_below_price(stock_symbol, max_price)
#     if result:
#         results_list.append(result)

# print("Stock predictions below 1500:")
# print(results_list)
import pandas as pd


max_price = 5000.0

results_list = []
stocks = ["ADANIPORTS.NS", 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS','BAJAJFINSV.NS','HEROMOTOCO.NS','ITC.NS',
          'BHARTIARTL.NS','WIPRO.NS','NESTLEIND.NS','HDFCBANK.NS','ZOMATO.NS','INFY.NS','MARUTI.NS','SBIN.NS',
          'ICICIBANK.NS','APOLLOHOSP.NS','TATASTEEL.NS','TCS.NS','TATAMOTORS.NS','BANKBARODA.NS','IDFCFIRSTB.NS','COLPAL.NS,'
            'DALBHARAT.NS','BATAINDIA.NS','ZYDUSWELL.NS','WHIRLPOOL.NS','RELIANCE.NS','HINDUNILVR.NS','PIDILITIND.NS','TITAN.NS',
          'TIINDIA.NS','JKCEMENT.NS','ADANIENT.NS','SKFINDIA.NS','EMAMILTD.NS','HINDZINC.NS','ADANIPOWER.NS']

for stock_symbol in stocks:
    result = fetch_stock_predictions_below_price(stock_symbol, max_price)
    if result:
        results_list.append(result)

df = pd.DataFrame(results_list)

print("Stock predictions below 1500:")
print(df)


df1 = df.to_json()
print(df1)
