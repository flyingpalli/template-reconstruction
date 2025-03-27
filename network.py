import logging
import os

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(
        self,
        input_dim=1071,
        output_dim=51,
        hidden_dim=256,
        num_layers=10,
        n_heads=8,
        kernel_size=50,
    ):
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = x.permute(2, 0, 1)
        x = self.transformer(x)

        x = x.mean(dim=0)
        x = self.fc(x)
        return x
