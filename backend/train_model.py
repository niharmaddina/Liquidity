# backend/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_definition import LNN_ODE_Model
from data_processing import get_historical_data, add_features, FEATURES
import joblib
import logging
import time


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def train_model():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    symbol = 'AAPL'
    data = add_features(get_historical_data(symbol))

    features = data[FEATURES].values
    targets = data['Close'].shift(-1).dropna().values
    features = features[:-1]  # Align features and targets

    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(
        1)  # Add dimension

    # Normalize features and targets
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Convert back to tensors and move to device
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=True, num_workers=0)

    input_size = 4
    hidden_size = 64
    output_size = 1
    model = LNN_ODE_Model(input_size, hidden_size, output_size).to(device)

    initialize_weights(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log every 10 batches
            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        end_time = time.time()
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}, Time: {end_time - start_time:.2f}s")

    torch.save(model.state_dict(), 'lnn_ode_model.pth')
    joblib.dump({'X': scaler_X, 'y': scaler_y}, 'scalers.joblib')
    logging.info("Model training complete; saved lnn_ode_model.pth and scalers.joblib.")


if __name__ == '__main__':
    train_model()
