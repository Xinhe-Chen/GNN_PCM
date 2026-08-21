"""Train an ANN surrogate model to predict day-ahead LMP (LMP DA).

Inputs (features):
    - Dispatch DA        (from data/PCM_results/sweep_200MW_PEM.csv)
    - RenewablesUsed      (from data/PCM_results/sweep_200MW_PEM.csv)
    - Load                (column '1' of data/GMLC_ts_data/DAY_AHEAD_regional_Load.csv)

Target:
    - LMP DA              (from data/PCM_results/sweep_200MW_PEM.csv)
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_PCM = REPO_ROOT / "data" / "PCM_results" / "sweep_base_case.csv"
PCM_CSV = REPO_ROOT / "data" / "PCM_results" / "sweep_200MW_PEM.csv"
LOAD_CSV = REPO_ROOT / "data" / "GMLC_ts_data" / "DAY_AHEAD_regional_Load.csv"

FEATURE_COLUMNS = ["Dispatch DA", "RenewablesUsed", "Load"]
TARGET_COLUMN = "LMP DA"

BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN_SIZES = (64, 64)
VAL_FRACTION = 0.2
RANDOM_SEED = 42
PLOT_PATH = pathlib.Path(__file__).resolve().parent / "val_predictions.png"


def load_dataset() -> pd.DataFrame:
    pcm_df = pd.read_csv(PCM_CSV)
    load_df = pd.read_csv(LOAD_CSV)
    base_df = pd.read_csv(BASE_PCM)

    if len(pcm_df) != len(load_df):
        raise ValueError(
            f"Row count mismatch between PCM data ({len(pcm_df)}) and load data "
            f"({len(load_df)}); expected them to be aligned hour-by-hour."
        )

    df = pcm_df[["Dispatch DA", "RenewablesUsed", "LMP DA"]].copy()
    df["Load"] = load_df["1"].to_numpy()
    df["base_LMP"] = base_df["LMP DA"].copy()
    return df


class ANNRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def plot_validation_results(y_true, y_pred, save_path=PLOT_PATH):
    """Scatter plot of predicted vs. actual LMP DA (original $/MWh scale)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Ideal (y = x)")

    ax.set_xlabel("Actual LMP DA ($/MWh)")
    ax.set_ylabel("Predicted LMP DA ($/MWh)")
    ax.set_title("Validation Set: Predicted vs. Actual LMP DA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved validation scatter plot to: {save_path}")


def main():
    torch.manual_seed(RANDOM_SEED)

    df = load_dataset()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32) - df["base_LMP"].to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED
    )

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train = x_scaler.transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).astype(np.float32).ravel()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32).ravel()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANNRegressor(input_size=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        train_loss = running_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = loss_fn(val_pred, y_val_t.to(device)).item()

        if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(
                f"Epoch {epoch:4d}/{NUM_EPOCHS} | "
                f"train MSE (scaled): {train_loss:.6f} | "
                f"val MSE (scaled): {val_loss:.6f}"
            )

    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(torch.from_numpy(X_train).to(device)).cpu().numpy()
        val_pred_scaled = model(X_val_t.to(device)).cpu().numpy()

    y_train_orig = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    train_pred_orig = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
    val_pred_orig = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()

    train_r2 = r2_score(y_train_orig, train_pred_orig)
    val_r2 = r2_score(y_val_orig, val_pred_orig)

    print("\nFinal results:")
    print(f"  Train MSE (scaled): {train_loss:.6f}")
    print(f"  Val   MSE (scaled): {val_loss:.6f}")
    print(f"  Train R2 (original scale): {train_r2:.4f}")
    print(f"  Val   R2 (original scale): {val_r2:.4f}")

    plot_validation_results(y_val_orig, val_pred_orig)


if __name__ == "__main__":
    main()
