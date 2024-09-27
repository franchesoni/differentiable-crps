import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Sequence
import tqdm
import schedulefree

import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    def __init__(self, time_series, valid_indices, context, h):
        self.time_series = time_series
        self.valid_indices = valid_indices
        self.context = context
        self.h = h

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_index = self.valid_indices[idx]
        input_indices = np.arange(
            target_index - self.context - self.h + 1, target_index - self.h + 1
        )
        input = torch.tensor(self.time_series[input_indices], dtype=torch.float32)
        target = torch.tensor(self.time_series[target_index], dtype=torch.float32)
        return input, target


class MLP(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], activation_fn=nn.GELU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn())
        self.network = nn.Sequential(*layers)
        self.norm_layer = (
            nn.Softmax(dim=1) if layer_sizes[-1] > 1 else nn.Identity()
        )  # softmax for prob distro

    def forward(self, x):
        return self.norm_layer(self.network(x))


def mae_loss(pred, target, **kwargs):
    return torch.mean(torch.abs(pred - target))


def binmass_to_cdf(binmass):
    return torch.cumsum(
        torch.cat(
            [torch.zeros(binmass.shape[0], 1).to(binmass.device), binmass], dim=1
        ),
        dim=1,
    )  # (num bin borders,)


def crps_loss_batch(binmass, y, bin_borders):
    """
    Compute CRPS loss for given bin borders and bin probability mass values.

    Args:
    binmass: tensor of shape (N, B)
    y: tensor of shape (N,)
    bin_borders: tensor of shape (B+1,)

    Returns:
    CRPS loss: tensor of shape (N,)
    """
    N, B = binmass.shape
    device = binmass.device

    # Ensure inputs are on the same device
    y = y.to(device)
    bin_borders = bin_borders.to(device)
    bin_widths = bin_borders[1:] - bin_borders[:-1]
    assert (0 <= bin_widths).all(), "bin_borders are not ordered."

    # Compute CDF from prob mass of bins
    cdf_values = torch.cat(
        [torch.zeros(N, 1, device=device), torch.cumsum(binmass, dim=1)], dim=1
    )

    # Compute parts
    parts = (cdf_values[:, :-1] + cdf_values[:, 1:]) / 2 * (bin_widths)
    sq_parts = (
        -1
        / 3
        * (
            cdf_values[:, :-1] ** 2
            + cdf_values[:, :-1] * cdf_values[:, 1:]
            + cdf_values[:, 1:] ** 2
        )
        * (-bin_widths)
    )

    # first part of the loss
    p1 = bin_borders[-1] - y
    # second part of the loss
    p2 = torch.sum(sq_parts, dim=1)

    # Find the bin index for each y
    purek = torch.searchsorted(bin_borders, y.view(-1, 1)) - 1
    k = torch.clamp(purek, 0, B - 1).squeeze()

    # Compute CDF at y using linear interpolation
    cdf_at_y = cdf_values[torch.arange(N), k] + (
        (
            (y - bin_borders[k])
            / (bin_widths[k])
            * (cdf_values[torch.arange(N), k + 1] - cdf_values[torch.arange(N), k])
        )
    )

    p3 = (
        (cdf_at_y + cdf_values[torch.arange(N), k + 1]) / 2 * (bin_borders[k + 1] - y)
    ) * ((bin_borders[0] - y < 0) * (y < bin_borders[-1])).float()
    mask = torch.arange(B, device=binmass.device)[None, :] > purek
    p4 = torch.sum(parts * mask, dim=1) * ((y - bin_borders[-1] < 0))

    crps = torch.abs(p1) + p2 - 2 * (p3 + p4)

    return crps


def memoize(func):
    cache = {}

    def memoized_func(*args):
        key = str(args)
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]

    return memoized_func


@memoize
def cached_linspace(start, end, steps):
    return torch.linspace(start, end, steps).to(DEVICE)


def crps_loss(pred, y, upper_bound):
    bin_borders = cached_linspace(0, upper_bound, 101)
    return torch.mean(crps_loss_batch(pred, y, bin_borders))


def ce_loss(pred, y, upper_bound):
    bin_borders = cached_linspace(0, upper_bound, 101)
    y_bin = torch.bucketize(y, bin_borders) - 1
    return nn.functional.nll_loss(torch.log(pred + 1e-8), y_bin)


def train(
    train_time_series,
    context=18,
    h=6,
    hidden_size=64,
    learning_rate=1e-3,
    num_steps=100,
    batch_size=32,
    val_every=10,
    num_layers=3,
    loss_name="mae",
    val_pct=0.1,
):

    # Data preparation
    series_len = len(train_time_series)
    valid_indices = np.arange(context + h - 1, series_len)
    val_size = int(val_pct * len(valid_indices))
    train_val_divisions = [
        val_size // 2,
        len(valid_indices) // 2,
        len(valid_indices) // 2 + val_size // 2,
    ]
    part1, part2, part3, part4 = np.split(valid_indices, train_val_divisions)
    valid_validation_indices = np.concatenate((part1, part3))
    valid_train_indices = np.concatenate((part2, part4))

    train_dataset = TimeSeriesDataset(
        train_time_series, valid_train_indices, context, h
    )
    val_dataset = TimeSeriesDataset(
        train_time_series, valid_validation_indices, context, h
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model initialization
    output_dim = 1 if loss_name == "mae" else 100
    model = MLP([context] + [hidden_size] * num_layers + [output_dim]).to(DEVICE)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=learning_rate, warmup_steps=int(num_steps * 0.05)
    )
    model.upper_bound = train_time_series.max() * (1 + 1 / len(train_time_series))

    if loss_name == "mae":
        loss_fn = mae_loss
    elif loss_name == "crps":
        loss_fn = crps_loss
    elif loss_name == "ce":
        loss_fn = ce_loss
    else:
        raise NotImplementedError("Loss function not yet implemented")

    # Training loop
    loss_curve = []
    val_loss_curve = {}
    step = 0
    while step < num_steps:
        model.train()
        optimizer.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y, upper_bound=model.upper_bound)
            loss.backward()
            optimizer.step()

            loss_curve.append(loss.item())

            if (step + 1) % val_every == 0:
                model.eval()
                optimizer.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X, val_y = val_X.to(DEVICE), val_y.to(DEVICE)
                        val_pred = model(val_X)
                        val_loss += loss_fn(
                            val_pred, val_y, upper_bound=model.upper_bound
                        ).item()
                val_loss /= len(val_loader)
                val_loss_curve[step] = val_loss
                print(
                    f"Step {step + 1}/{num_steps}, Train Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}",
                    end="\r",
                )
                if val_loss_curve[step] == min(val_loss_curve.values()):
                    torch.save(model.state_dict(), f"best_val_model.pth")
                model.train()
                optimizer.train()

            step += 1
            if step >= num_steps:
                break
    model.load_state_dict(torch.load("best_val_model.pth"))
    model.eval()
    optimizer.eval()
    return model, loss_curve, val_loss_curve


def evaluate(test_time_series, model, context, h, metric_name="diff"):
    model.eval()
    errors = []
    bin_borders = cached_linspace(0, model.upper_bound, 101)
    with torch.no_grad():
        for target_index in tqdm.tqdm(range(context + h - 1, len(test_time_series))):
            input_indices = np.arange(
                target_index - context - h + 1, target_index - h + 1
            )
            input = (
                torch.tensor(test_time_series[input_indices], dtype=torch.float32)
                .unsqueeze(0)
                .to(DEVICE)
            )
            target = torch.tensor(
                test_time_series[target_index], dtype=torch.float32
            ).to(DEVICE)
            pred = model(input)
            if metric_name == "diff":
                if pred.numel() > 1:
                    # assume pred is a prob dist over bins
                    cdf = binmass_to_cdf(pred).squeeze()  # should be (101,) now
                    # find the index where 0.5 should be inserted to maintain order
                    i50p = torch.searchsorted(cdf, 0.5)
                    i50m = i50p - 1
                    # find the value corresponding to 0.5
                    pred = bin_borders[i50m] + (
                        bin_borders[i50p] - bin_borders[i50m]
                    ) * (0.5 - cdf[i50m]) / (cdf[i50p] - cdf[i50m])
                error = pred - target
            elif metric_name == "crps":
                if pred.numel() == 1:
                    bin_borders = torch.tensor(
                        [0, pred.item(), pred.item() + 1e-9, model.upper_bound]
                    ).to(DEVICE)
                    pred = torch.tensor([[0, 1, 0]]).to(DEVICE)
                error = crps_loss_batch(pred, target[None], bin_borders).mean()
            elif metric_name == "ce":
                error = ce_loss(pred[None], target[None], model.upper_bound).mean()
            else:
                raise NotImplementedError("only diff was implemented")
            errors.append(error.item())
    return errors


def main(
    plot=False,
    b=False,
    context=18,
    h=6,
    size=["s", "m", "l"],
    learning_rate=[
        1e-5 * 4**i for i in range(6)
    ],  # 5e-5 is ok for mae and 1e-3 for crps
    batch_size=[32, 64, 128],
    num_steps=500,
    val_every=10,
    loss_name="mae",
    seed=0,
):
    assert loss_name in ["mae", "ce", "crps"]

    # read the data
    datapath = "LE_E10_012016_122017_C10x10_fil.csv"
    data = pd.read_csv(datapath, header=None)

    # data preparation
    csi = data[9].values  # clear sky index
    train_time_series = csi[: len(csi) // 2]
    test_time_series = csi[
        len(csi) // 2 + 6 * 24 :
    ]  # leave one day ofintervals space between train and test (10min freq explains 6 * 24 it's a day)

    best_val_loss = np.inf
    exp_number = 0
    for s in size:
        if s == "s":
            hidden_size = 32
            num_layers = 1
        elif s == "m":
            hidden_size = 128
            num_layers = 3
        elif s == "l":
            hidden_size = 512
            num_layers = 5
        for lr in learning_rate:
            for bs in batch_size:
                # seed everything
                np.random.seed(seed)
                torch.manual_seed(seed)
                # train the model
                model, loss_curve, val_loss_curve = train(
                    train_time_series,
                    context,
                    h,
                    hidden_size,
                    lr,
                    num_steps,
                    bs,
                    val_every,
                    num_layers,
                    loss_name,
                )
                if min(val_loss_curve.values()) < best_val_loss:
                    best_val_loss = min(val_loss_curve.values())
                    best_model = model
                    best_so_far = True
                else:
                    best_so_far = False
                print("=" * 80)
                print("Experiment number", exp_number)
                print("Parameters:", loss_name, "size", s, "lr", lr, "batch size", bs)
                print("Best validation loss:", best_val_loss)
                print("Best so far:", best_so_far)
                exp_number += 1

    print("Evaluating best model...")
    model = best_model

    # evaluate the best model
    errors = evaluate(test_time_series, model, context, h, metric_name="diff")
    errors = np.array(errors).flatten()
    print(f"Test MAE:", np.mean(np.abs(errors)))

    crpss = evaluate(test_time_series, model, context, h, metric_name="crps")
    crpss = np.array(crpss).flatten()
    print(f"Test CRPS:", np.mean(crpss))

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.log(loss_curve), label="Training")
        plt.plot(
            list(val_loss_curve.keys()),
            np.log(list(val_loss_curve.values())),
            label="Validation",
        )
        plt.xlabel("Step")
        plt.ylabel("Log Loss")
        plt.legend()
        plt.savefig(f"figs/log_curve_{loss_name}.png")

        plt.figure()
        plt.hist(errors, bins=1000)
        plt.xlabel("Errors")
        plt.ylabel("Count")
        plt.savefig(f"figs/hist_{loss_name}.png")

    if b:
        import ipdb

        ipdb.set_trace()


# RESULTS:

# Parameters: mae size s lr 0.00016 batch size 32
# Best validation loss: 0.17201702264802796
# Test MAE: 0.1547732080386128
# Test CRPS: 0.15477322783504066

# Parameters: ce size l lr 0.00256 batch size 64
# Best validation loss: 3.6224256311144147
# Test MAE: 0.17074449098917552
# Test CRPS: 0.1279946840045208

# Parameters: crps size m lr 0.00256 batch size 128
# Best validation loss: 0.1059756154815356
# Test MAE: 0.12988348985547576
# Test CRPS: 0.09746930291549602

# Now running with 
#                  `python main.py --plot --size="[m,]" --learning_rate="[0.00256,]" --batch_size="[128,]" --loss_name=mae`
#                  `python main.py --plot --size="[m,]" --learning_rate="[0.00256,]" --batch_size="[128,]" --loss_name=ce`
#                  `python main.py --plot --size="[m,]" --learning_rate="[0.00256,]" --batch_size="[128,]" --loss_name=crps`
# (these are the best hparams for CRPS in validation, and CE with these params outperforms the params chosen by CE val loss)
# ================================================================================
# Experiment number 0
# Parameters: mae size m lr 0.00256 batch size 128
# Best validation loss: 0.2461947219239341
# Best so far: True
# Evaluating best model...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22043/22043 [00:03<00:00, 6248.20it/s]
# Test MAE: 0.1573037923251696
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22043/22043 [00:13<00:00, 1628.51it/s]
# Test CRPS: 0.1573037959646487
# ================================================================================
# Experiment number 0
# Parameters: ce size m lr 0.00256 batch size 128
# Best validation loss: 3.6347153584162393
# Best so far: True
# Evaluating best model...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22043/22043 [00:05<00:00, 3725.66it/s]
# Test MAE: 0.13040203570118658
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22043/22043 [00:12<00:00, 1758.34it/s]
# Test CRPS: 0.09755296096694463
# ================================================================================
# Experiment number 0
# Parameters: crps size m lr 0.00256 batch size 128
# Best validation loss: 0.10597561899986532
# Best so far: True
# Evaluating best model...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22043/22043 [00:05<00:00, 3844.65it/s]
# Test MAE: 0.12988349102116056
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22043/22043 [00:12<00:00, 1785.49it/s]
# Test CRPS: 0.09746930379835754


if __name__ == "__main__":
    # start a grid search over the parameters

    from fire import Fire

    Fire(main)

