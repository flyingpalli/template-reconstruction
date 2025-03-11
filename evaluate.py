import argparse
import logging
import os

import numpy as np
import pandas
import torch
import torch.nn as nn


def read_spec(path: os.PathLike):
    df = pandas.read_csv(path, sep="\r", header=None, dtype=np.float64).transpose()
    return torch.tensor(df.values)


def write_losvd(y: torch.Tensor, path: os.PathLike):
    res = [f"{str(val)}\n" for val in y.tolist()[0]]
    with open(path, mode="w") as f:
        f.writelines(res)


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument(
            "--model",
            type=str,
            default=None,
            required=True,
            help="Path of the model",
        )
        self.parser.add_argument(
            "--spec",
            type=str,
            default=None,
            required=True,
            help="Path of the spectrum for evaluation",
        )
        self.parser.add_argument(
            "--losvd",
            type=str,
            default=None,
            required=False,
            help="Path of the resulting losvd",
        )
        self.parser.add_argument(
            "--kernel",
            type=int,
            default=50,
            help="Size of the convolution kernel in the convolution layers",
        )
        self.parser.add_argument(
            "--num-layers",
            type=int,
            default=10,
            help="Number of layers in the network",
        )
        self.parser.add_argument(
            "--hidden-dim",
            type=int,
            default=256,
            help="Dimension of the hidden layers in the network",
        )

    def parse_args(self):
        return self.parser.parse_args()


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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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


def load_model_all(state_dict: dict, path: str, device) -> None:
    """
    Load all models and optimizers from a checkpoint file
    :param state_dict: The state dictionary to load the models and optimizers into
    :param path: The path to the model
    :param device: The torch device
    :return:
    """
    assert os.path.exists(path), "Path to model does not exist"
    checkpoint = torch.load(path, device)
    for name, obj in state_dict.items():
        if name in checkpoint:
            obj.load_state_dict(checkpoint[name])
    logging.info(f"Models and optimizers loaded from {path}")


def main():
    parser = ArgParser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level="INFO",
    )

    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    torch.set_default_device(device)
    torch.multiprocessing.set_start_method("spawn")
    logging.info(f"Using device: {device}")

    network = Network(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel,
    ).to(device)

    load_model_all({"model": network}, path=args.model, device=device)

    x = read_spec(args.spec)
    y = network(x.to(device))

    losvd = args.losvd
    if losvd is None:
        losvd = f"{args.spec}_eval"

    write_losvd(y, losvd)


if __name__ == "__main__":
    main()
