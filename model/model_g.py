import torch
from torch import nn, Tensor

class G_RNN(nn.Module):
    def __init__(self, input_dim: int, category_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(category_dim + input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(category_dim + input_dim + hidden_dim, output_dim)
        self.o2o = nn.Linear(hidden_dim + output_dim, output_dim)
        self.dropout = nn.Dropout(0.1) 
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

    def forward(self, c: Tensor, x: Tensor, h: Tensor) -> Tensor:
        combined = torch.cat((c, x, h), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.o2o(torch.cat((hidden, output), 1))
        output = self.dropout(output)
        return output, hidden