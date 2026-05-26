import torch
import torch.nn as nn


class EmotionMLP(nn.Module):
    def __init__(self, input_dim=772, hidden_dim=256, num_blocks=3, output_dim=2, dropout_p=0.2):
        super(EmotionMLP, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            ))

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.bn_in(x)
        x = self.relu(x)

        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual

        x = self.output_proj(x)
        return x
