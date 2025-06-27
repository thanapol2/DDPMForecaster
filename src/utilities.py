import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

def longest_valid_run(series):
    mask = series.notnull().to_numpy()
    max_len = 0
    curr_len = 0
    for valid in mask:
        if valid:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0
    return max_len

def mc_dropout_predict(model, context, num_samples=100):
    model.train()  # Important: keep dropout ON at inference
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(context).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)



def create_windows_from_panel(data, input_len, forecast_len):
    # data: (M, N)
    X, Y = [], []
    M, N = data.shape
    for i in range(N - input_len - forecast_len + 1):
        X.append(data[:, i:i+input_len])
        Y.append(data[:, i+input_len:i+input_len+forecast_len])
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y


def plot_diffusion_forecasts(
        y_true,  # (N, forecast_len)
        pred_samples,  # (N, num_mc, forecast_len)
        prediction_length,
        prediction_intervals=(50.0, 90.0),
        color="g",
        fname=None,
        rows=4,
        cols=4
):
    N, forecast_len = y_true.shape
    num_mc = pred_samples.shape[1]
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    index = np.arange(forecast_len)

    # Compute percentiles for each interval
    percentiles = []
    for interval in prediction_intervals:
        p_low = 50.0 - interval / 2.0
        p_high = 50.0 + interval / 2.0
        percentiles.extend([p_low, 50.0, p_high])
    percentiles = sorted(set(percentiles))  # unique and sorted
    percentiles = [p for p in percentiles if 0 <= p <= 100]

    for i in range(min(N, rows * cols)):
        ax = axx[i]
        # Ground truth
        ax.plot(index, y_true[i], color="k", label="observations")

        # Compute required percentiles across MC samples
        qs = np.percentile(pred_samples[i], percentiles, axis=0)  # shape: (len(percentiles), forecast_len)

        # Median
        median_idx = percentiles.index(50.0)
        ax.plot(index, qs[median_idx], color=color, linestyle="-", label="median prediction")

        # Prediction intervals (fill between)
        interval_labels = []
        for interval in prediction_intervals:
            low_idx = percentiles.index(50.0 - interval / 2.0)
            high_idx = percentiles.index(50.0 + interval / 2.0)
            ax.fill_between(
                index,
                qs[low_idx],
                qs[high_idx],
                color=color,
                alpha=0.2 if interval == max(prediction_intervals) else 0.4,
                label=f"{int(interval)}% prediction interval" if i == 0 else None
            )

        ax.set_title(f"Series {i + 1}")
        ax.legend()

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def plot_single_forecast(
        y_true,  # (forecast_len,)
        pred_samples,  # (num_mc, forecast_len)
        prediction_intervals=(50.0, 90.0),
        color="g",
        title="Forecast",
        fname=None
):
    forecast_len = y_true.shape[0]
    index = np.arange(forecast_len)

    # Compute percentiles for the desired intervals
    percentiles = []
    for interval in prediction_intervals:
        p_low = 50.0 - interval / 2.0
        p_high = 50.0 + interval / 2.0
        percentiles.extend([p_low, 50.0, p_high])
    percentiles = sorted(set(percentiles))
    percentiles = [p for p in percentiles if 0 <= p <= 100]

    qs = np.percentile(pred_samples, percentiles, axis=0)  # (len(percentiles), forecast_len)
    median_idx = percentiles.index(50.0)

    plt.figure(figsize=(12, 7))
    plt.plot(index, y_true, color="k", label="observations")
    plt.plot(index, qs[median_idx], color=color, linestyle="-", label="median prediction")

    # Plot prediction intervals
    for interval in prediction_intervals:
        low_idx = percentiles.index(50.0 - interval / 2.0)
        high_idx = percentiles.index(50.0 + interval / 2.0)
        plt.fill_between(
            index,
            qs[low_idx],
            qs[high_idx],
            color=color,
            alpha=0.2 if interval == max(prediction_intervals) else 0.4,
            label=f"{int(interval)}% prediction interval"
        )
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)
    plt.show()


# ----- Metrics -----
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    errors = np.abs(y_true - y_pred).mean(axis=1)
    return errors.mean()

def mase(y_true, y_pred, y_train, m=1):
    d = np.abs(y_train[:, m:] - y_train[:, :-m]).mean(axis=1)
    errors = np.abs(y_true - y_pred).mean(axis=1)
    return (errors / d).mean()

def mape(y_true, y_pred):
    return (np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))).mean() * 100

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    return (np.abs(y_true - y_pred) / denominator).mean() * 200

def crps_ensemble(samples, y_true):
    M, F = samples.shape[1], samples.shape[2]
    crps_all = np.zeros((M, F))
    for m in range(M):
        for t in range(F):
            x = samples[:, m, t]   # (num_samples,)
            y = y_true[m, t]
            term1 = np.mean(np.abs(x - y))
            term2 = 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
            crps_all[m, t] = term1 - term2
    return crps_all.mean()

class MultiTSDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

