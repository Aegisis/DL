import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

stock_symbol = 'AAPL'
df = yf.download(stock_symbol, start="2010-01-01", end="2023-01-01")
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

look_back = 60
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

def create_dataset(dataset, look_back=1):
    X = np.array([dataset[i:i+look_back, 0] for i in range(len(dataset)-look_back)])
    y = np.array([dataset[i+look_back, 0] for i in range(len(dataset)-look_back)])
    return X.reshape(-1, look_back, 1), y

X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

def process_model(layer_type):
    # Build and train model
    model = Sequential([layer_type(50, input_shape=(look_back, 1)), Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    test_pred = scaler.inverse_transform(model.predict(X_test))

    last_seq = X_test[-1].copy()
    future_preds = []
    for _ in range(30):
        pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)[0, 0]
        future_preds.append(pred)
        last_seq = np.append(last_seq[1:], pred)

    return test_pred, scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

lstm_pred, lstm_future = process_model(LSTM)
gru_pred, gru_future = process_model(GRU)
y_actual = scaler.inverse_transform(y_test.reshape(1, -1))

plt.figure(figsize=(14,7))
future_range = np.arange(len(y_actual[0]), len(y_actual[0]) + 30)

plt.plot(y_actual[0], label='Actual Price', color='blue')
plt.plot(lstm_pred, label='LSTM Predictions', color='orange')
plt.plot(gru_pred, label='GRU Predictions', color='green')
plt.plot(future_range, lstm_future, label='Future LSTM', color='red', linestyle='--')
plt.plot(future_range, gru_future, label='Future GRU', color='purple', linestyle='--')

plt.title(f'{stock_symbol} Stock price prediction - LSTM vs GRU')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
