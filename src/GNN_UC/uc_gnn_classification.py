"""
GNN node-classification model for unit commitment (UC) status prediction,
following the node-classification formulation in:

    "A Graph Neural Network Based Framework for Unit Commitment Status
    Prediction" (https://ieeexplore.ieee.org/abstract/document/10246391)

Each generator is a graph node. Nodes are connected through the underlying
transmission topology (two generators are linked if their host buses are
joined by a line) plus a "same-bus" coupling between generators that sit on
the same bus. Given a day's 24-hour nodal demand profile as the node input
feature -- optionally concatenated with the host bus's 24-hour available
renewable generation profile (`--include-renewable`, on by default) -- the
model predicts each generator's 24-hour on/off (1/0) commitment status as a
multi-label node classification problem.

The trained model is saved as a self-contained bundle (weights + constructor
config + graph + feature normalization statistics) under model/GNN_with_renewable;
see `save_model` / `load_model`.

Data source: src/GNN_UC/data_prep.py (topology, node/edge features, and
generator commitment status extracted from the GMLC/PCM simulation data).
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

import data_prep

MODEL_DIR = Path(__file__).resolve().parents[2] / "model" / "GNN_with_renewable"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_generator_graph(dataset: dict, same_bus_weight: float = 1.0):
    """Build the generator-level graph from the bus topology.

    Two generators are linked if:
      - their host buses are joined by a transmission line (edge weight =
        that line's susceptance, from EF), or
      - they sit on the same bus (edge weight = `same_bus_weight`, modeling
        the shared bus power-balance coupling).

    Returns
    -------
    edge_index : torch.LongTensor [2, 2*E']
    edge_weight : torch.FloatTensor [2*E']
    """
    gen_bus_idx = dataset["gen_bus_idx"]  # [G], bus node index per generator
    bus_edge_index = dataset["edge_index"]  # [2, E] bus-level line list
    susceptance = dataset["EF"][:, 0]  # [E]

    bus_to_gens = {}
    for g, b in enumerate(gen_bus_idx):
        bus_to_gens.setdefault(int(b), []).append(g)

    src, dst, weight = [], [], []

    # Line-induced generator-generator edges.
    for e in range(bus_edge_index.shape[1]):
        bus_a, bus_b = int(bus_edge_index[0, e]), int(bus_edge_index[1, e])
        gens_a = bus_to_gens.get(bus_a, [])
        gens_b = bus_to_gens.get(bus_b, [])
        for ga in gens_a:
            for gb in gens_b:
                src += [ga, gb]
                dst += [gb, ga]
                weight += [susceptance[e], susceptance[e]]

    # Same-bus generator-generator edges.
    for gens in bus_to_gens.values():
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                ga, gb = gens[i], gens[j]
                src += [ga, gb]
                dst += [gb, ga]
                weight += [same_bus_weight, same_bus_weight]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float)
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Dataset: one graph per simulated day
# ---------------------------------------------------------------------------

def build_daily_graphs(dataset: dict, edge_index: torch.Tensor, edge_weight: torch.Tensor):
    """Build one `torch_geometric.data.Data` graph per simulated day.

    Node input feature (x): the 24-hour demand profile of the generator's
    host bus for that day, shape [G, 24]. When the dataset was built with
    `include_renewable=True`, the host bus's 24-hour available renewable
    profile is concatenated as a second channel block, giving [G, 48]:
    columns 0-23 are demand, columns 24-47 are renewables.
    Node label (y): the generator's 24-hour on/off commitment status for
    that day, shape [G, 24].
    """
    NF = dataset["NF"]  # [N, 24, D]
    commitment = dataset["commitment"]  # [G, 24, D]
    gen_bus_idx = dataset["gen_bus_idx"]  # [G]
    NR = dataset.get("NR") if dataset.get("has_renewable") else None  # [N, 24, D] or None

    n_days = commitment.shape[2]
    graphs = []
    for d in range(n_days):
        demand = NF[gen_bus_idx, :, d]  # [G, 24]
        feats = demand if NR is None else np.concatenate([demand, NR[gen_bus_idx, :, d]], axis=1)
        x = torch.tensor(feats, dtype=torch.float)  # [G, 24] or [G, 48]
        y = torch.tensor(commitment[:, :, d], dtype=torch.float)  # [G, 24]
        graphs.append(Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight))
    return graphs


def split_graphs(graphs: list, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    n_train = int(train_frac * len(graphs))
    n_val = int(val_frac * len(graphs))

    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]
    return train, val, test


def normalize_features(train: list, val: list, test: list):
    """Z-score node features using train-set statistics only."""
    x_all = torch.cat([g.x for g in train], dim=0)
    mean, std = x_all.mean(dim=0, keepdim=True), x_all.std(dim=0, keepdim=True).clamp_min(1e-6)
    for split in (train, val, test):
        for g in split:
            g.x = (g.x - mean) / std
    return mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UCGNN(nn.Module):
    """Multi-layer GCN node-classification model.

    Input: node feature x [num_nodes, T] (day-ahead 24-hour bus demand).
    Output: node logits [num_nodes, T] (per-hour on/off commitment).
    """

    def __init__(self, in_channels: int = 24, hidden_channels: int = 64, out_channels: int = 24, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)  # logits, shape [num_nodes, T]


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    out_dir: Path,
    model: "UCGNN",
    model_config: dict,
    norm_stats: tuple,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    metrics: dict,
    extra: dict | None = None,
):
    """Save a self-contained checkpoint plus human-readable metadata.

    The `.pt` bundle carries everything needed to rebuild the model and run
    inference correctly: weights, the constructor config, the graph the model
    was trained on, and the train-set feature normalization statistics.
    Persisting the normalization is essential -- applying the model to raw,
    unnormalized features silently produces wrong predictions.

    A sidecar `metadata.json` records the same config/metrics in readable form
    for experiment tracking (it is not read back by `load_model`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = norm_stats
    checkpoint = {
        "format_version": 1,
        "model_class": "UCGNN",
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "feature_mean": mean.cpu(),
        "feature_std": std.cpu(),
        "edge_index": edge_index.cpu(),
        "edge_weight": edge_weight.cpu(),
        "metrics": metrics,
        "torch_version": torch.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        checkpoint.update(extra)

    model_path = out_dir / "uc_gnn_with_renewable.pt"
    torch.save(checkpoint, model_path)

    metadata = {
        "model_class": "UCGNN",
        "model_config": model_config,
        "metrics": metrics,
        "torch_version": torch.__version__,
        "saved_at": checkpoint["saved_at"],
        "checkpoint_file": model_path.name,
    }
    if extra:
        metadata.update({k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool, list, dict))})
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path


def load_model(path: Path, device=None):
    """Rebuild a trained UCGNN from a `save_model` checkpoint.

    Returns
    -------
    model : UCGNN, in eval mode
    checkpoint : dict
        The full bundle, including "feature_mean"/"feature_std" (needed to
        normalize inputs the same way as at training time) and the graph
        "edge_index"/"edge_weight".
    """
    path = Path(path)
    if path.is_dir():
        path = path / "uc_gnn_with_renewable.pt"

    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = UCGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_weight)
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
    parser = argparse.ArgumentParser(description="Train a GNN node-classification model for UC status prediction.")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-renewable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate the host bus's hourly available renewable profile onto the node features.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory to save the trained model bundle into.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = data_prep.build_dataset(
        hours=None, commitment_hours=None, include_renewable=args.include_renewable
    )
    edge_index, edge_weight = build_generator_graph(dataset)
    graphs = build_daily_graphs(dataset, edge_index, edge_weight)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs, seed=args.seed)
    norm_stats = normalize_features(train_graphs, val_graphs, test_graphs)

    # Derived from the data so the demand-only (24) and demand+renewable (48)
    # cases both work without editing the model definition.
    in_channels = graphs[0].x.shape[1]
    out_channels = graphs[0].y.shape[1]
    print(f"node input features: {in_channels} (renewables {'on' if args.include_renewable else 'off'})")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model_config = {
        "in_channels": in_channels,
        "hidden_channels": args.hidden_channels,
        "out_channels": out_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = UCGNN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, best_state, best_epoch = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, device)
    print(f"best val acc {best_val_acc:.4f} | test loss {test_loss:.4f} test acc {test_acc:.4f}")

    model_path = save_model(
        out_dir=args.out_dir,
        model=model,
        model_config=model_config,
        norm_stats=norm_stats,
        edge_index=edge_index,
        edge_weight=edge_weight,
        metrics={
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
        },
        extra={
            "include_renewable": args.include_renewable,
            "feature_layout": (
                "columns 0-23: host-bus hourly demand; columns 24-47: host-bus hourly available renewables"
                if args.include_renewable
                else "columns 0-23: host-bus hourly demand"
            ),
            "generator_ids": dataset["generator_ids"].tolist(),
            "training_args": vars(args),
        },
    )
    print(f"saved model bundle to {model_path}")


if __name__ == "__main__":
    main()
