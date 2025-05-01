import torch
import torch.nn as nn

class PredictiveEmotion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.linear(h[-1])
