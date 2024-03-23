import torch
from torch import nn, Tensor
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        combined = torch.cat((x, h), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.rnn(x, h0)
#         return self.fc(out[:, -1, :])