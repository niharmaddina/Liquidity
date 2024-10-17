# backend/model_definition.py
import torch
import torch.nn as nn
from torchdiffeq import odeint

class LNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LNNCell, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.lambda_param = nn.Parameter(torch.ones(hidden_size))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x, h_prev):
        dh_dt = -self.lambda_param * h_prev + torch.tanh(
            x @ self.W.t() + h_prev @ self.U.t() + self.bias
        )
        h = h_prev + dh_dt
        return h

class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t, h):
        return self.net(h)

class LNN_ODE_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LNN_ODE_Model, self).__init__()
        self.hidden_size = hidden_size
        self.lnn_cell = LNNCell(input_size, hidden_size)
        self.ode_func = ODEFunc(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_size)
        h = self.lnn_cell(x, h0)
        t = torch.linspace(0, 1, steps=2)
        h_t = odeint(self.ode_func, h, t)[-1]
        output = self.fc(h_t)
        return output
