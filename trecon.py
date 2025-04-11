import logging
import os
import re
import time

import duckdb
import numpy as np
import pandas
import tomli
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from torch.utils.data import DataLoader

import network

app = typer.Typer()


def load_config():
    with open("config.toml", "rb") as fb:
        return tomli.load(fb)


def read_spec(path: os.PathLike):
    df = pandas.read_csv(path, sep="\r", header=None, dtype=np.float64).transpose()
    return torch.tensor(df.values)


def write_losvd(y: torch.Tensor, path: os.PathLike):
    res = [f"{str(val)}\n" for val in y.tolist()[0]]
    with open(path, mode="w") as f:
        f.writelines(res)


@app.command()
def evaluate(
    model_path: str, spec: str, hidden_dim: int = 256, num_layers: int = 10, kernel_size: int = 50, losvd: str = ""
):
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

    model = network.Network(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
    ).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    x = read_spec(spec)
    y = model(x.to(device))

    losvd = losvd
    if not losvd:
        losvd = f"{spec}_eval"

    write_losvd(y, losvd)


@app.command()
def train(
    hidden_dim: int = 256,
    num_layers: int = 10,
    kernel_size: int = 50,
    lr: float = 1e-5,
    model_path: str = "",
    batch_size: int = 1,
):
    config = load_config()

    run_name = f"tempRecon__{int(time.time())}"
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=config["program"]["log_level"].upper(),
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

        model = network.Network(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        file_starts = 0
        if model_path:
            match = re.match(r".*tempRecon__\d+/(\d+)/.*", model_path)
            if match:
                step = int(match.group(1))
                file_starts = step // config["data"]["file_size"]
            else:
                logging.warning("Could not deduce step from path. Starting from zero.")
            network.load_model_all({"model": model, "optimizer": optimizer}, path=model_path, device=device)
            logging.info(f"Models and optimizers loaded from {model_path}")

        for file in range(file_starts, 20):
            spectrum = duckdb.read_csv(f"{config['data']['spectrum']}_{file}").df().values
            losvd = duckdb.read_csv(f"{config['data']['losvd']}_{file}").df().values
            dataloader = DataLoader(
                dataset=list(zip(spectrum, losvd)),
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                generator=torch.Generator(device=device),
            )
            for i, (x, target) in enumerate(dataloader):
                current_step = config["data"]["file_size"] * file + i
                optimizer.zero_grad()
                pred = model(x.to(device))
                loss = loss_f(pred, target.to(device))
                logging.info(f"Step {current_step}, Loss: {loss:.2f}")
                loss.backward()
                optimizer.step()
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught! Saving progress...")
    finally:
        network.save_model_all(
            run_name,
            {
                "model_state_dict": model,
                "optimizer_state_dict": optimizer,
            },
        )


if __name__ == "__main__":
    app()
