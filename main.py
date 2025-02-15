import os.path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Network(nn.Module):
    def __init__(
        self,
        input_dim=1071,
        output_dim=51,
        hidden_dim=256,
        num_layers=5,
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
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

        # self.fc_first = nn.Linear(input_dim, hidden_dim)
        # self.hidden_layers = nn.ModuleList()
        # for _ in range(n_layers):
        #     self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # self.fc_last = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.dropout(x)

        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = x.permute(2, 0, 1)
        x = self.transformer(x)

        x = x.mean(dim=0)
        x = self.fc(x)
        return x

        # x = F.sigmoid(self.fc_first(x))
        # for layer in self.hidden_layers:
        #     x = F.sigmoid(layer(x))
        # x = self.fc_last(x)
        # return x


if __name__ == "__main__":
    BATCH_SIZE = 1
    LR = 1e-5
    LOSVD = "./data/losvd_split/losvd_scaled"
    SPECTRUM = "./data/spec_split/spec"

    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    torch.set_default_device(device)
    torch.multiprocessing.set_start_method("spawn")
    print("Using device:", device)

    loss_f = nn.MSELoss(reduction="sum")

    network = Network().to(device)
    optimizer = optim.Adam(network.parameters(), lr=LR)

    for file in range(20):
        spectrum = duckdb.read_csv(f"{SPECTRUM}_{file}").df().values
        losvd = duckdb.read_csv(f"{LOSVD}_{file}").df().values
        dataloader = DataLoader(
            dataset=list(zip(spectrum, losvd)),
            batch_size=BATCH_SIZE,
            num_workers=1,
            shuffle=True,
            generator=torch.Generator(device=device),
        )
        losses = 0
        for i, (x, target) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = network(x.to(device))
            loss = loss_f(pred, target.to(device))
            loss.backward()
            optimizer.step()
            print(" step:", file * 50000 + i, "    loss:", loss.item())
            # losses += loss.item()
            # if i % 100 == 0:
            #     print(" step:", file * 50000 + i, "    loss:", losses / 100)
            #     losses = 0
