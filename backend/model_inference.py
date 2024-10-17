# backend/model_inference.py
import torch
from model_definition import LNN_ODE_Model

def load_model(model_path):
    input_size = 4  # Number of input features
    hidden_size = 64
    output_size = 1
    model = LNN_ODE_Model(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def make_prediction(model, data):
    input_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    prediction = output.item()
    return prediction
