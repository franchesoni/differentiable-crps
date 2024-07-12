# [DONE] ensure that we're not overfitting by evaluating on a validation set during training
# [DONE] implement the crps loss
# [DONE] implement the ce loss
# implement the reliability diagram
# check that the reliability diagram improves with more iterations
# check that the mae of the median of the predicted distribution improves with more iterations
# compare against the mae obtained by training with the mae
# compare closed form crps against the numerical integral, check that the integral approximates the crps loss with more and more points


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Sequence
import tqdm
import schedulefree


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


def pdf_to_cdf(pdf):
    return torch.cumsum(
        torch.cat([torch.zeros(pdf.shape[0], 1).to(pdf.device), pdf], dim=1),
        dim=1,
    )


def crps_loss_fixed_bins(predicted_pdf, y, bin_borders):
    """
    Compute CRPS loss for fixed bin borders and predicted CDF values.
    """
    device = y.device
    bin_borders = bin_borders.to(device)
    batch_size = y.shape[0]

    predicted_cdf = pdf_to_cdf(predicted_pdf)

    j = torch.searchsorted(bin_borders, y) - 1
    j = torch.clamp(j, 0, len(bin_borders) - 2)

    F_y = predicted_cdf[torch.arange(batch_size), j] + (
        predicted_cdf[torch.arange(batch_size), j + 1]
        - predicted_cdf[torch.arange(batch_size), j]
    ) / (bin_borders[j + 1] - bin_borders[j]) * (y - bin_borders[j])

    def integral_F_squared(F_i, F_i_plus_1, b_i, b_i_plus_1):
        return -1 / 3 * (F_i**2 + F_i * F_i_plus_1 + F_i_plus_1**2) * (b_i - b_i_plus_1)

    def integral_1_minus_F_squared(F_i, F_i_plus_1, b_i, b_i_plus_1):
        return (
            -1
            / 3
            * (3 + F_i**2 + F_i * (-3 + F_i_plus_1) - 3 * F_i_plus_1 + F_i_plus_1**2)
            * (b_i - b_i_plus_1)
        )

    # Create masks for each sample
    mask = torch.arange(len(bin_borders) - 1, device=device)[None, :] < j[:, None]

    # Compute the first part of CRPS
    crps_part1 = torch.sum(
        integral_F_squared(
            predicted_cdf[:, :-1] * mask,
            predicted_cdf[:, 1:] * mask,
            bin_borders[:-1].expand(batch_size, -1) * mask,
            bin_borders[1:].expand(batch_size, -1) * mask,
        ),
        dim=1,
    )

    # Compute the second part of CRPS
    crps_part2 = integral_F_squared(
        predicted_cdf[torch.arange(batch_size), j], F_y, bin_borders[j], y
    )

    # Compute the third part of CRPS
    crps_part3 = integral_1_minus_F_squared(
        F_y, predicted_cdf[torch.arange(batch_size), j + 1], y, bin_borders[j + 1]
    )

    # Compute the fourth part of CRPS
    mask_inv = ~mask
    crps_part4 = torch.sum(
        integral_1_minus_F_squared(
            predicted_cdf[:, :-1] * mask_inv,
            predicted_cdf[:, 1:] * mask_inv,
            bin_borders[:-1].expand(batch_size, -1) * mask_inv,
            bin_borders[1:].expand(batch_size, -1) * mask_inv,
        ),
        dim=1,
    )

    crps = crps_part1 + crps_part2 + crps_part3 + crps_part4

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
    return torch.mean(crps_loss_fixed_bins(pred, y, bin_borders))


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

    if loss_name == "mae":
        loss_fn = mae_loss
    elif loss_name == "crps":
        loss_fn = crps_loss
    elif loss_name == "ce":
        loss_fn = ce_loss
    else:
        raise NotImplementedError("Loss function not yet implemented")

    model.upper_bound = train_time_series.max() * (1 + 1 / len(train_time_series))
    loss_curve = []
    val_loss_curve = {}

    # Training loop
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
                    f"Step {step + 1}/{num_steps}, Train Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}"
                )
                model.train()
                optimizer.train()
            print(
                f"Step {step + 1}/{num_steps}, Train Loss: {loss.item():.4f}", end="\r"
            )

            step += 1
            if step >= num_steps:
                break
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
                    cdf = pdf_to_cdf(pred).squeeze()  # should be (101,) now
                    # find the index where 0.5 should be inserted to maintain order
                    i50p = torch.searchsorted(cdf, 0.5)
                    i50m = i50p - 1
                    # find the value corresponding to 0.5
                    pred = bin_borders[i50m] + (
                        bin_borders[i50p] - bin_borders[i50m]
                    ) * (0.5 - cdf[i50m]) / (cdf[i50p] - cdf[i50m])
                error = pred - target
            elif metric_name == "crps":
                error = crps_loss_fixed_bins(
                    pred[None], target[None], bin_borders
                ).mean()
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
    hidden_size=128,
    num_layers=3,
    learning_rate=1e-4,  # 5e-5 is ok for mae and 1e-3 for crps
    num_steps=100,
    batch_size=64,
    val_every=10,
    loss_name="mae",
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert loss_name in ["mae", "ce", "crps"]
    # read the data
    datapath = "LE_E10_012016_122017_C10x10_fil.csv"
    data = pd.read_csv(datapath, header=None)

    # data preparation
    csi = data[9].values  # clear sky index
    train_time_series = csi[: len(csi) // 2]
    test_time_series = csi[
        len(csi) // 2 + 144 :
    ]  # leave one day of space between train and test (10min intervals)

    # train the model
    model, loss_curve, val_loss_curve = train(
        train_time_series,
        context,
        h,
        hidden_size,
        learning_rate,
        num_steps,
        batch_size,
        val_every,
        num_layers,
        loss_name,
    )

    # evaluate the model
    errors = evaluate(test_time_series, model, context, h, metric_name="diff")
    errors = np.array(errors).flatten()

    print(f"Test MAE:", np.mean(np.abs(errors)))

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

        plt.figure()
        plt.hist(errors, bins=1000)
        plt.xlabel("Errors")
        plt.ylabel("Count")
        plt.show()

    if b:
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
