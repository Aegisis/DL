import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.random.seed(42)
torch.manual_seed(42)

time_steps = np.linspace(0, 100, 400)
prices = np.sin(time_steps) + np.random.normal(scale=0.5, size=len(time_steps))

plt.figure(figsize=(10, 6))
plt.plot(time_steps, prices, label='Stock Price')
plt.title('Simulated Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

scaler = MinMaxScaler(feature_range=(-1, 1))
prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))

seq_length = 10
x, y = [], []
for i in range(len(prices_normalized) - seq_length):
    x.append(prices_normalized[i:i+seq_length])
    y.append(prices_normalized[i+seq_length])
x, y = np.array(x), np.array(y)

train_size = int(len(y) * 0.67)
trainX = torch.Tensor(x[:train_size]).view(-1, seq_length, 1).to(device)
trainY = torch.Tensor(y[:train_size]).to(device)
testX = torch.Tensor(x[train_size:]).view(-1, seq_length, 1).to(device)
testY = torch.Tensor(y[train_size:]).to(device)
fullX = torch.Tensor(x).view(-1, seq_length, 1).to(device)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

model = LSTM(1, 128, 3, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler_amp = torch.cuda.amp.GradScaler()

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        outputs = model(trainX)
        loss = criterion(outputs, trainY)

    scaler_amp.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler_amp.step(optimizer)
    scaler_amp.update()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/500], Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    full_predict = model(fullX)

full_predict = scaler.inverse_transform(full_predict.cpu().numpy())
actual_prices = scaler.inverse_transform(y)

plt.figure(figsize=(10, 6))
plt.axvline(x=train_size, c='r', linestyle='--', label='Train/Test Split')
plt.plot(actual_prices, label='Actual Price', color='blue')
plt.plot(full_predict, label='Predicted Price', color='orange')
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()