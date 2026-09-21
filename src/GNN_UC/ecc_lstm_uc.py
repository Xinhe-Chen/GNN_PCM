"""
Spatio-temporal ECC + LSTM architecture for unit commitment node classification.

Reproduces the architecture of Fig. 4 in
https://ieeexplore.ieee.org/abstract/document/10246391:

    Inputs                ECC layer 1        ECC layer 2        LSTM layer            FC + sigmoid
    node: N x Nfeatures   node act: PReLU    node act: PReLU    seq length: T         Output: P(n=1)
    edge: E x Efeatures   edge act: PReLU    edge act: PReLU    act: tanh             shape: N x T
                                                                recurrent act: sigmoid

ECC (Edge-Conditioned Convolution, Simonovsky & Komodakis 2017) is provided by
`torch_geometric.nn.NNConv`: an edge network maps each edge's feature vector to
the weights of the filter applied to that neighbor's message, so the line
susceptance / thermal rating directly condition how information propagates.

The LSTM then sweeps the T-hour horizon per node, and a shared fully-connected
layer maps each hour's hidden state to the commitment logit P(unit on) for that
node-hour, giving the N x T output of the figure.

Node granularity, feature/label construction, and data sources follow
`uc_gnn_classification.py` / `data_prep.py`: one node per generator, node input
= the host bus's 24-hour demand profile, node label = the generator's 24-hour
on/off commitment status.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv

import data_prep
from uc_gnn_classification import split_graphs


# ---------------------------------------------------------------------------
# Graph construction (edge features, not just scalar weights)
# ---------------------------------------------------------------------------

def build_generator_graph(dataset: dict, same_bus_features=(0.0, 0.0)):
    """Build the generator-level graph with per-edge feature vectors.

    Two generators are linked if their host buses are joined by a
    transmission line, or if they sit on the same bus. Line-induced edges
    carry that line's EF row (susceptance, thermal limit); same-bus edges
    carry `same_bus_features` as a sentinel, so the ECC edge network can
    learn to treat the two couplings differently.

    Returns
    -------
    edge_index : torch.LongTensor [2, E']
    edge_attr : torch.FloatTensor [E', 2]
    """
    gen_bus_idx = dataset["gen_bus_idx"]  # [G], bus node index per generator
    bus_edge_index = dataset["edge_index"]  # [2, E] bus-level line list
    EF = dataset["EF"]  # [E, 2] susceptance, thermal limit

    bus_to_gens: dict[int, list[int]] = {}
    for g, b in enumerate(gen_bus_idx):
        bus_to_gens.setdefault(int(b), []).append(g)

    src, dst, attr = [], [], []

    # Line-induced generator-generator edges (both directions).
    for e in range(bus_edge_index.shape[1]):
        bus_a, bus_b = int(bus_edge_index[0, e]), int(bus_edge_index[1, e])
        for ga in bus_to_gens.get(bus_a, []):
            for gb in bus_to_gens.get(bus_b, []):
                src += [ga, gb]
                dst += [gb, ga]
                attr += [EF[e], EF[e]]

    # Same-bus generator-generator edges (both directions).
    for gens in bus_to_gens.values():
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                ga, gb = gens[i], gens[j]
                src += [ga, gb]
                dst += [gb, ga]
                attr += [same_bus_features, same_bus_features]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.asarray(attr, dtype=np.float32), dtype=torch.float)
    return edge_index, edge_attr


def build_daily_graphs(dataset: dict, edge_index: torch.Tensor, edge_attr: torch.Tensor):
    """Build one `Data` graph per simulated day.

    x: [G, T] host-bus demand profile for that day.
    y: [G, T] generator on/off commitment status for that day.
    """
    NF = dataset["NF"]  # [N, T, D_days]
    commitment = dataset["commitment"]  # [G, T, D_samples]
    gen_bus_idx = dataset["gen_bus_idx"]  # [G]
    NR = dataset.get("NR") if dataset.get("has_renewable") else None

    n_samples = commitment.shape[2]
    # With merged PCM runs, commitment is indexed by sample while NF/NR stay
    # indexed by calendar day; "sample_day" maps between the two.
    sample_day = dataset.get("sample_day")
    if sample_day is None:
        sample_day = np.arange(n_samples)
    sample_run = dataset.get("sample_run", np.full(n_samples, "unknown"))

    graphs = []
    for s in range(n_samples):
        d = int(sample_day[s])
        demand = NF[gen_bus_idx, :, d]
        feats = demand if NR is None else np.concatenate([demand, NR[gen_bus_idx, :, d]], axis=1)
        x = torch.tensor(feats, dtype=torch.float)
        y = torch.tensor(commitment[:, :, s], dtype=torch.float)
        graphs.append(
            Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
                day_index=int(d),
                run_name=str(sample_run[s]),
            )
        )
    return graphs


def normalize_features(train: list, val: list, test: list):
    """Z-score node and edge features using train-set statistics only."""
    x_all = torch.cat([g.x for g in train], dim=0)
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp_min(1e-6)

    # Edge attributes are shared across days, so one graph's copy suffices.
    e_all = train[0].edge_attr
    e_mean = e_all.mean(dim=0, keepdim=True)
    e_std = e_all.std(dim=0, keepdim=True).clamp_min(1e-6)
    edge_attr_norm = (e_all - e_mean) / e_std

    for split in (train, val, test):
        for g in split:
            g.x = (g.x - x_mean) / x_std
            g.edge_attr = edge_attr_norm

    return (x_mean, x_std), (e_mean, e_std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EdgeNet(nn.Module):
    """Edge network for an ECC layer: edge features -> filter weights.

    Maps [E, num_edge_features] to [E, in_channels * out_channels], with the
    PReLU "edge activation" of the figure.
    """

    def __init__(self, num_edge_features: int, in_channels: int, out_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_edge_features, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
            nn.PReLU(),
            nn.Linear(hidden, in_channels * out_channels),
        )

    def forward(self, edge_attr):
        return self.net(edge_attr)


class ECCLSTM(nn.Module):
    """ECC x2 -> LSTM -> fully-connected sigmoid, per Fig. 4.

    Parameters
    ----------
    num_node_features : int
        Nfeatures of the input node matrix (here T, the 24-hour demand profile).
    num_edge_features : int
        Efeatures of the input edge matrix (here 2: susceptance, thermal limit).
    horizon : int
        T, the sequence length fed to the LSTM and the output width.
    ecc_hidden : int
        Output channels of ECC layer 1.
    lstm_in_channels : int
        Per-timestep channel count produced by ECC layer 2; ECC layer 2 emits
        `horizon * lstm_in_channels` features per node, reshaped to [N, T, C].
    lstm_hidden : int
        LSTM hidden size (its output channels).

    Forward returns logits of shape [N, T]; apply a sigmoid (or use
    `predict_proba`) to get P(n=1) as drawn in the figure.
    """

    def __init__(
        self,
        num_node_features: int = 24,
        num_edge_features: int = 2,
        horizon: int = 24,
        ecc_hidden: int = 64,
        lstm_in_channels: int = 16,
        lstm_hidden: int = 64,
        edge_net_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm_in_channels = lstm_in_channels
        self.dropout = dropout

        ecc2_out = horizon * lstm_in_channels

        # ECC layer 1 + PReLU node activation.
        self.ecc1 = NNConv(
            num_node_features,
            ecc_hidden,
            EdgeNet(num_edge_features, num_node_features, ecc_hidden, edge_net_hidden),
            aggr="mean",
        )
        self.act1 = nn.PReLU()

        # ECC layer 2 + PReLU node activation.
        self.ecc2 = NNConv(
            ecc_hidden,
            ecc2_out,
            EdgeNet(num_edge_features, ecc_hidden, ecc2_out, edge_net_hidden),
            aggr="mean",
        )
        self.act2 = nn.PReLU()

        # LSTM over the T-hour horizon. PyTorch's LSTM already uses tanh for the
        # cell/output activation and sigmoid for the input/forget/output gates,
        # matching the "Activation: Tanh / Recurrent activation: Sigmoid" note.
        self.lstm = nn.LSTM(lstm_in_channels, lstm_hidden, batch_first=True)

        # Shared fully-connected head applied at every timestep -> N x T logits.
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        # Spatial encoding: two edge-conditioned convolutions.
        h = self.act1(self.ecc1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.act2(self.ecc2(h, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Temporal decoding: unfold node embedding into a length-T sequence.
        # Nodes are the LSTM "batch", so this works unchanged for PyG mini-batches.
        h = h.view(-1, self.horizon, self.lstm_in_channels)  # [N, T, C]
        h, _ = self.lstm(h)  # [N, T, lstm_hidden]

        return self.fc(h).squeeze(-1)  # [N, T] logits

    @torch.no_grad()
    def predict_proba(self, x, edge_index, edge_attr):
        """P(n=1) for every node-hour, shape [N, T]."""
        return torch.sigmoid(self.forward(x, edge_index, edge_attr))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_commitment(model, graphs, device=None, threshold: float = 0.5, batch_size: int = 16):
    """Predict generator commitment from a trained ECC+LSTM model.

    Parameters
    ----------
    model : ECCLSTM
        A trained model. Put into eval mode here, so dropout is disabled.
    graphs : Data | sequence of Data
        GNN input built by `build_daily_graphs` (one graph per day), already
        normalized the same way as training. A single `Data` is accepted and
        returns un-batched [G, T] arrays.
    device : torch.device, optional
        Defaults to the device the model's parameters already live on.
    threshold : float
        P(n=1) above which a unit is declared on.
    batch_size : int
        Mini-batch size used for the forward passes.

    Returns
    -------
    dict with keys
        "commitment"    : np.ndarray [D, G, T] of 0/1 (or [G, T] for one graph)
        "probabilities" : np.ndarray of the same shape, P(unit on)
        "accuracy"      : float, element-wise accuracy against the graphs'
                          ground-truth labels; only present when every input
                          graph carries a `y`.

    Notes
    -----
    Assumes all input graphs share the same node count, which holds for the
    fixed generator-level topology used throughout this module.
    """
    single = isinstance(graphs, Data)
    graphs = [graphs] if single else list(graphs)
    if not graphs:
        raise ValueError("`graphs` is empty; nothing to predict.")

    n_nodes = graphs[0].num_nodes
    if any(g.num_nodes != n_nodes for g in graphs):
        raise ValueError("All input graphs must have the same number of nodes.")

    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    prob_chunks, label_chunks = [], []
    # shuffle=False keeps the output rows aligned with the input day order.
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)  # [B*G, T]
        probs = torch.sigmoid(logits).view(batch.num_graphs, n_nodes, -1)  # [B, G, T]
        prob_chunks.append(probs.cpu())
        if batch.y is not None:
            label_chunks.append(batch.y.view(batch.num_graphs, n_nodes, -1).cpu())

    probabilities = torch.cat(prob_chunks, dim=0)
    commitment = (probabilities > threshold).to(torch.int8)

    result = {}
    if len(label_chunks) == len(prob_chunks):
        labels = torch.cat(label_chunks, dim=0)
        result["accuracy"] = (commitment.float() == labels).float().mean().item()

    probabilities = probabilities.numpy()
    commitment = commitment.numpy()
    if single:
        probabilities, commitment = probabilities[0], commitment[0]

    result["commitment"] = commitment
    result["probabilities"] = probabilities
    return result


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == batch.y).sum().item()
        total_count += batch.y.numel()
        total_loss += loss.item() * batch.y.numel()

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser(description="Train the ECC+LSTM UC node-classification model (Fig. 4).")
    parser.add_argument("--ecc-hidden", type=int, default=64)
    parser.add_argument("--lstm-in-channels", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--edge-net-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pcm-runs",
        nargs="+",
        default=["all"],
        help=(
            "PCM result folders under data/PCM_results to draw commitment labels from. "
            "'all' (default) merges every available run; or name them explicitly."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).with_name("ecc_lstm_uc_checkpoint.pt")))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pcm_runs = "all" if args.pcm_runs == ["all"] else args.pcm_runs
    dataset = data_prep.build_dataset(hours=None, commitment_hours=None, pcm_runs=pcm_runs)
    print(f"PCM runs merged: {dataset['pcm_runs']} | samples: {dataset['commitment'].shape[2]}")
    edge_index, edge_attr = build_generator_graph(dataset)
    graphs = build_daily_graphs(dataset, edge_index, edge_attr)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs, seed=args.seed)
    normalize_features(train_graphs, val_graphs, test_graphs)

    horizon = graphs[0].y.shape[1]
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model = ECCLSTM(
        num_node_features=graphs[0].x.shape[1],
        num_edge_features=edge_attr.shape[1],
        horizon=horizon,
        ecc_hidden=args.ecc_hidden,
        lstm_in_channels=args.lstm_in_channels,
        lstm_hidden=args.lstm_hidden,
        edge_net_hidden=args.edge_net_hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

    model.load_state_dict(torch.load(args.checkpoint))
    test_loss, test_acc = run_epoch(model, test_loader, device)
    print(f"best val acc {best_val_acc:.4f} | test loss {test_loss:.4f} test acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
