# backend/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_definition import LNN_ODE_Model
from data_processing import get_historical_data, preprocess_data

def train_model():
    symbol = 'AAPL'
    data = get_historical_data(symbol)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    features = data[['Close', 'MA20', 'MA50', 'Return']].values
    targets = data['Close'].shift(-1).dropna().values  # Predict next day's close
    features = features[:-1]  # Align features and targets
    targets = targets.reshape(-1, 1)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = 4
    hidden_size = 64
    output_size = 1
    model = LNN_ODE_Model(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

    torch.save(model.state_dict(), 'lnn_ode_model.pth')

if __name__ == '__main__':
    train_model()
