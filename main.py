import argparse
import logging
import os
import re
import time

import duckdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batchsize for supervised learning",
        )
        self.parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of the optimizer")
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
        self.parser.add_argument(
            "--losvd",
            type=str,
            default="./data/losvd_split/losvd_scaled",
            help="Directory which contains the losvd files",
        )
        self.parser.add_argument(
            "--spectrum",
            type=str,
            default="./data/spec_split/spec",
            help="Directory which contains the spec files",
        )
        self.parser.add_argument("--log-level", type=str, default="INFO", help="Level of the console logger")
        self.parser.add_argument("--file-size", type=int, default=50000, help="Amount of lines per file")
        self.parser.add_argument(
            "--model",
            type=str,
            default="",
            help="Path of the model, when pre-loading is desired",
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
            batch_first=True,
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


def save_model_all(run_name: str, step: int, state_dict: dict) -> None:
    """
    Save all models and optimizers to a checkpoint file
    :param run_name: The name of the run
    :param step: The current step
    :param state_dict: The current state dictionary
    :return:
    """
    model_folder = f"./models/{run_name}/{step}"
    os.makedirs(model_folder, exist_ok=True)
    out_path = f"{model_folder}/checkpoint.pth"

    total_state_dict = {name: obj.state_dict() for name, obj in state_dict.items()}
    torch.save(total_state_dict, out_path)
    logging.info(f"Saved all models and optimizers to {out_path}")


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

    run_name = f"tempRecon__{int(time.time())}"
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=args.log_level.upper(),
    )

    try:
        torch.set_default_dtype(torch.float64)
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True
        else:
            device = "cpu"

        torch.set_default_device(device)
        torch.multiprocessing.set_start_method("spawn")
        logging.info(f"Using device: {device}")

        loss_f = nn.MSELoss(reduction="sum")

        network = Network(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel,
        ).to(device)
        optimizer = optim.Adam(network.parameters(), lr=args.lr)

        file_starts = 0
        if args.model:
            match = re.match(r".*tempRecon__\d+/(\d+)/.*", args.model)
            if match:
                step = int(match.group(1))
                file_starts = step // args.file_size
                load_model_all({"model": network}, path=args.model, device=device)
            else:
                logging.error("Could not deduce step from path. Starting from zero.")

        for file in range(file_starts, 20):
            spectrum = duckdb.read_csv(f"{args.spectrum}_{file}").df().values
            losvd = duckdb.read_csv(f"{args.losvd}_{file}").df().values
            dataloader = DataLoader(
                dataset=list(zip(spectrum, losvd)),
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
                generator=torch.Generator(device=device),
            )
            losses = 0
            for i, (x, target) in enumerate(dataloader):
                current_step = args.file_size * file + i
                optimizer.zero_grad()
                pred = network(x.to(device))
                loss = loss_f(pred, target.to(device))
                losses += loss
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    logging.info(f"Step {current_step}, Loss: {losses / 10:.2f}")
                    losses = 0
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught! Saving progress...")
    finally:
        state_dict = {"model": network}
        save_model_all(run_name, current_step, state_dict)


if __name__ == "__main__":
    main()
