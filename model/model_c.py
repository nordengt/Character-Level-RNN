import torch
from torch import nn, Tensor

class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        combined = torch.cat((x, h), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    
class T_RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h