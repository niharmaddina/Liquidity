# backend/model_inference.py
import torch
import joblib
from model_definition import LNN_ODE_Model

def load_model(model_path):
    input_size = 4  # Number of input features
    hidden_size = 64
    output_size = 1
    model = LNN_ODE_Model(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_scalers(path='scalers.joblib'):
    return joblib.load(path)


def make_prediction(model, data, scalers):
    """`data` is one raw feature row (1 x 4); returns the predicted next-day close in price units."""
    x = torch.tensor(scalers['X'].transform(data), dtype=torch.float32)
    with torch.no_grad():
        output = model(x).numpy()
    return float(scalers['y'].inverse_transform(output)[0, 0])
