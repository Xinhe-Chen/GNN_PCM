"""
Train and validate a GNN surrogate for the DC economic dispatch of a
Prescient production-cost-model run on RTS-GMLC.

The model follows the graph-filter formulation of

    Owerko, Gama & Ribeiro, "Optimal Power Flow Using Graph Neural Networks",
    ICASSP 2020

specialized to the DC case. Each layer is a polynomial graph filter

    Y = sigma( Sum_{k=0}^{K-1} S^k X H_k )

with S the normalized susceptance matrix from data_prep. The parameter count
Sum_l F_{l-1} F_l K_l is independent of the number of buses, which is the
property that motivates a GNN here at all; the fully-connected baseline in this
file scales as N * F per layer and is printed alongside for comparison.

Filters are applied densely (S is 73x73) rather than through torch_geometric.
At this size a dense matmul is faster than sparse message passing, and a
polynomial filter has no per-edge parameters for a PyG conv to carry, so the
extra dependency would buy nothing.

Why a fully-connected baseline is not optional
----------------------------------------------
Falconer & Mones (IEEE TPWRS 2022) found that on fixed topology an FCNN matches
GNNs across grids of 24-2853 buses, including the 73-bus RTS case this dataset
is built on. Without the baseline a good GNN number says nothing about whether
the graph structure contributed. `--model both` (the default) trains and
reports both.

Why the metrics are stratified
------------------------------
Two properties of this dataset make a single pooled error number misleading:

  * ~88% of hours have zero LMP spread across buses, i.e. no binding line. In
    those hours DC-OPF degenerates to a merit-order sort and the network is
    irrelevant. The remaining ~12% carry an average spread of ~$42/MWh, larger
    than the mean LMP itself. Metrics are therefore reported separately for
    congested and uncongested hours.
  * ~55% of committed unit-hours sit exactly at PMin or PMax, where the value
    is pinned by commitment and capacity -- both of which are *input features*.
    A model can score well there without learning anything about dispatch, so
    the interior (marginal) buses are reported separately.

Physics recovery
----------------
Branch flows are not a second output head. Under DC assumptions they are exact
in the injections, f = PTDF @ p, and data_prep validates that identity against
the PCM's own flows to 0.09 MW. A learned flow head could only approximate a
matrix we already hold in closed form, and could contradict its own dispatch
prediction. So flows -- and the congestion metrics built on them -- are derived
from the predicted dispatch, and the same PTDF can optionally enter the loss as
a line-overload penalty (`--flow-penalty`).

Usage
-----
    python data_prep.py                  # build data/model/opf_dcopf_dataset.npz
    python train_OPF_GNN.py              # train GNN + FCNN baseline
    python train_OPF_GNN.py --flow-penalty 0.1 --epochs 400
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

import data_prep

MODEL_DIR = Path(__file__).resolve().parents[2] / "model" / "GNN_OPF"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphFilterLayer(nn.Module):
    """One polynomial graph filter: Y = Sum_{k=0}^{K-1} S^k X H_k.

    The powers S^0..S^{K-1} are precomputed once and registered as a buffer of
    shape [K, N, N], so a forward pass is a single einsum. K is the filter's
    reach in hops: K=3 lets a bus see its two-hop neighbourhood, which on the
    73-bus RTS graph already spans a large fraction of each area.
    """

    def __init__(self, shift: torch.Tensor, in_features: int, out_features: int, taps: int, bias: bool = True):
        super().__init__()
        if taps < 1:
            raise ValueError("taps (K) must be >= 1")

        n = shift.shape[0]
        powers = [torch.eye(n, dtype=shift.dtype)]
        for _ in range(taps - 1):
            powers.append(powers[-1] @ shift)
        self.register_buffer("powers", torch.stack(powers))  # [K, N, N]

        self.weight = nn.Parameter(torch.empty(taps, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Scale by fan-in across all taps together; initializing each tap as if
        # it were the only one makes the summed output variance grow with K.
        nn.init.normal_(self.weight, std=1.0 / np.sqrt(taps * in_features))

    def forward(self, x):  # x: [B, N, F_in] -> [B, N, F_out]
        shifted = torch.einsum("knm,bmf->bknf", self.powers, x)
        out = torch.einsum("bknf,kfg->bng", shifted, self.weight)
        return out if self.bias is None else out + self.bias


class OPFGNN(nn.Module):
    """Graph-filter network scoring each generator bus for dispatch allocation.

    Readout modes mirror the two variants of the reference paper:

      "local"  a node-wise linear map, then gather the generator buses. Every
               operation is reachable through neighbour exchanges only, so the
               model stays decentralized and its parameter count is
               independent of N.
      "global" flatten all N nodes and map to the M outputs with one dense
               layer. More expressive, but the readout alone is N*F_L*M
               parameters and the locality property is lost -- Falconer & Mones
               attribute most of the predictive power of "local" GNNs in the
               literature to exactly this layer, so it is worth reporting which
               one produced a given number.

    The output is an unnormalized score per generator bus, not a dispatch
    level: `project_dispatch` turns the scores into MW through a softmax
    allocation of the system total. Regressing MW directly would let the ~75%
    of label entries that are exactly zero (nothing committed at that bus)
    dominate the loss, and would leave the system total to be learned rather
    than imposed.
    """

    def __init__(
        self,
        shift: np.ndarray,
        gen_node_idx: np.ndarray,
        in_features: int,
        hidden_features: tuple[int, ...] = (32, 16),
        taps: int = 3,
        readout: str = "local",
        dropout: float = 0.0,
    ):
        super().__init__()
        if readout not in ("local", "global"):
            raise ValueError(f"readout must be 'local' or 'global', got {readout!r}")

        shift_t = torch.as_tensor(shift, dtype=torch.float32)
        self.register_buffer("gen_node_idx", torch.as_tensor(gen_node_idx, dtype=torch.long))
        self.readout_mode = readout
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        dims = (in_features,) + tuple(hidden_features)
        self.filters = nn.ModuleList(
            GraphFilterLayer(shift_t, dims[i], dims[i + 1], taps) for i in range(len(hidden_features))
        )

        n_nodes, n_out = shift_t.shape[0], len(gen_node_idx)
        if readout == "local":
            self.readout = nn.Linear(dims[-1], 1)
        else:
            self.readout = nn.Linear(n_nodes * dims[-1], n_out)

    def forward(self, x):  # x: [B, N, F_in] -> [B, M]
        for f in self.filters:
            x = self.dropout(self.act(f(x)))
        if self.readout_mode == "local":
            return self.readout(x).squeeze(-1)[:, self.gen_node_idx]
        return self.readout(x.flatten(start_dim=1))


class OPFMLP(nn.Module):
    """Fully-connected baseline: flatten every bus feature, map to M outputs.

    Topology-agnostic by construction, so any gap between this and OPFGNN is
    attributable to the graph structure. Its first layer alone is N*F_in*H
    parameters, which is what the GNN's N-independent count is meant to avoid.
    """

    def __init__(self, n_nodes: int, in_features: int, n_out: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.0):
        super().__init__()
        layers, prev = [], n_nodes * in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


# ---------------------------------------------------------------------------
# Physics recovery
# ---------------------------------------------------------------------------

def required_total(fixed: torch.Tensor):
    """System thermal generation implied by the load and renewable output.

    DC power flow is lossless, so the injections must sum to zero:
    sum(thermal) = -sum(net_injection_fixed). This is computable from the
    inputs at inference time, which makes it a free hard constraint rather
    than something the network has to learn.

    It is exact up to ~5 MW: the two RTS-GMLC units (ROR, STORAGE) that
    Prescient omits from its detail files leave the observed hourly balance
    short by that much on average.
    """
    return -fixed.sum(dim=1, keepdim=True)


def project_dispatch(logits: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor, target_total: torch.Tensor | None = None, iters: int = 3):
    """Turn raw network scores into a feasible, power-balanced dispatch.

    The network predicts an unnormalized *preference* per bus, not a dispatch
    level. The dispatch is then constructed so that the physics holds by
    construction:

        p_i = PMin_i + (T - sum PMin) * softmax(logits)_i ,  clipped to PMax_i

    where T is the system total from `required_total`. This mirrors the
    structure of economic dispatch itself: the total is fixed by load and
    renewables, and the only real question is who produces it. Fixing the total
    analytically removes a ~337 MW system-level error that appeared when the M
    bus outputs were predicted independently -- per-bus errors accumulate
    across 28 buses instead of cancelling.

    Why softmax and not a clamp. An earlier version predicted a capacity
    fraction and hard-clamped it to [0, 1]. Because a ReLU network starts near
    zero, every bus landed on the low side of the clamp, where the gradient is
    exactly zero; the balance correction then produced the same
    capacity-proportional allocation regardless of the network's output, and
    the GNN and the MLP trained to byte-identical results. Softmax keeps the
    gradient alive at every bus while still guaranteeing the allocation sums to
    one.

    Buses that hit PMax have their surplus redistributed over the buses with
    remaining headroom; `iters` passes handle cascades. If the committed
    minimums already exceed T -- possible in a high-renewable hour -- the
    minimums are scaled down proportionally, since no feasible allocation
    exists and the alternative is emitting negative dispatch.

    Everything here is differentiable, so it sits inside the training loop and
    the network learns against the projected output rather than fighting it at
    inference time.
    """
    zeros = torch.zeros_like(pmax)
    committed = pmax > 0

    if target_total is None:
        # Diagnostic path only (`--no-balance`): treat the scores as capacity
        # fractions through a sigmoid, which at least keeps gradients finite.
        mw = torch.clamp(torch.sigmoid(logits) * pmax, min=pmin, max=pmax)
        return torch.where(committed, mw, zeros)

    masked = torch.where(committed, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))
    weight = torch.where(committed, torch.softmax(masked, dim=1), zeros)

    base = torch.where(committed, pmin, zeros)
    base_total = base.sum(dim=1, keepdim=True)
    surplus = target_total - base_total

    p = torch.clamp(base + torch.clamp(surplus, min=0.0) * weight, max=pmax)
    for _ in range(iters):
        gap = (base_total + torch.clamp(surplus, min=0.0)) - p.sum(dim=1, keepdim=True)
        room = torch.where(committed, torch.clamp(pmax - p, min=0.0), zeros)
        share = room / room.sum(dim=1, keepdim=True).clamp_min(1e-6)
        p = torch.clamp(p + gap * share, max=pmax)

    infeasible_scale = torch.clamp(target_total / base_total.clamp_min(1e-6), min=0.0, max=1.0)
    p = torch.where(surplus < 0, base * infeasible_scale, p)
    return torch.where(committed, p, zeros)


def branch_flows(dispatch: torch.Tensor, fixed: torch.Tensor, ptdf: torch.Tensor, gen_idx: torch.Tensor):
    """Derive branch flows [B, L] from a dispatch, via the exact DC map f = PTDF @ p."""
    injection = fixed.clone()
    injection.index_add_(1, gen_idx, dispatch)
    return injection @ ptdf.T


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _masked_mae(err: np.ndarray, mask: np.ndarray):
    return float(np.abs(err[mask]).mean()) if mask.any() else float("nan")


def evaluate(model, tensors: dict, ptdf: torch.Tensor, device, balance: bool = True):
    """Compute dispatch, flow and congestion metrics on one split.

    Returned keys:
      dispatch_mae_mw            over committed buses
      dispatch_mae_mw_interior   committed buses strictly inside [PMin, PMax]
      dispatch_mae_mw_at_bound   committed buses pinned at a bound
      dispatch_mae_mw_congested / _uncongested
      total_dispatch_mae_mw      error in system-wide thermal generation
      flow_mae_mw                against the PCM's reported flows
      congestion_accuracy        agreement on which lines are >= 99% of rating
      congestion_f1              F1 on the same call; accuracy alone is
                                 uninformative when only ~0.2% of line-hours
                                 are at a limit
      balance_residual_mw        mean |sum of injections|; ~0 when the balance
                                 projection is on, which is the point of it
    """
    model.eval()
    with torch.no_grad():
        fixed = tensors["fixed"].to(device)
        pmin, pmax = tensors["pmin"].to(device), tensors["pmax"].to(device)
        pred_mw = project_dispatch(model(tensors["X"].to(device)), pmin, pmax, required_total(fixed) if balance else None)
        flows = branch_flows(pred_mw, fixed, ptdf.to(device), tensors["gen_idx"].to(device))
        injection = fixed.clone()
        injection.index_add_(1, tensors["gen_idx"].to(device), pred_mw)

    pred = pred_mw.cpu().numpy()
    true = tensors["y_mw"].numpy()
    pmin_np, pmax_np = tensors["pmin"].numpy(), tensors["pmax"].numpy()
    err = pred - true

    committed = pmax_np > 0
    tol = 0.5
    at_bound = committed & ((true >= pmax_np - tol) | (true <= pmin_np + tol))
    interior = committed & ~at_bound

    congested = tensors["congested"].numpy().astype(bool)
    hour_mask = np.broadcast_to(congested[:, None], committed.shape)

    flows_np = flows.cpu().numpy()
    flows_ref = tensors["flows_ref"].numpy()
    valid = np.isfinite(flows_ref)
    ratings = tensors["line_ratings"].numpy()[None, :]
    pred_binding = np.abs(flows_np) >= 0.99 * ratings
    true_binding = np.abs(flows_ref) >= 0.99 * ratings

    tp = float((pred_binding & true_binding & valid).sum())
    fp = float((pred_binding & ~true_binding & valid).sum())
    fn = float((~pred_binding & true_binding & valid).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "dispatch_mae_mw": _masked_mae(err, committed),
        "dispatch_mae_mw_interior": _masked_mae(err, interior),
        "dispatch_mae_mw_at_bound": _masked_mae(err, at_bound),
        "dispatch_mae_mw_congested": _masked_mae(err, committed & hour_mask),
        "dispatch_mae_mw_uncongested": _masked_mae(err, committed & ~hour_mask),
        "total_dispatch_mae_mw": float(np.abs(pred.sum(1) - true.sum(1)).mean()),
        "flow_mae_mw": float(np.abs(flows_np[valid] - flows_ref[valid]).mean()),
        "congestion_accuracy": float(((pred_binding == true_binding) & valid).sum() / valid.sum()),
        "congestion_precision": precision,
        "congestion_recall": recall,
        "congestion_f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "balance_residual_mw": float(np.abs(injection.cpu().numpy().sum(axis=1)).mean()),
        "n_hours": int(len(true)),
        "n_hours_congested": int(congested.sum()),
    }


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def make_split_tensors(data: dict, mask: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray):
    """Slice the dataset to one split and z-score its features.

    Normalization statistics are always the training split's. The static
    channels (incident line capacity) are constant over time, so their std is
    computed across buses; `clamp_min` in the caller keeps a zero-variance
    channel from producing NaNs.
    """
    X = (data["X"][mask] - feature_mean) / feature_std
    return {
        "X": torch.as_tensor(X, dtype=torch.float32),
        "y_frac": torch.as_tensor(data["y_frac"][mask], dtype=torch.float32),
        "y_mw": torch.as_tensor(data["y"][mask], dtype=torch.float32),
        "pmin": torch.as_tensor(data["committed_pmin"][mask], dtype=torch.float32),
        "pmax": torch.as_tensor(data["committed_pmax"][mask], dtype=torch.float32),
        "fixed": torch.as_tensor(data["net_injection_fixed"][mask], dtype=torch.float32),
        "flows_ref": torch.as_tensor(data["flows_ref"][mask], dtype=torch.float32),
        "line_ratings": torch.as_tensor(data["line_ratings"], dtype=torch.float32),
        "congested": torch.as_tensor(data["lmp_spread"][mask] > 1.0),
        "gen_idx": torch.as_tensor(data["gen_node_idx"], dtype=torch.long),
    }


def feature_stats(X: np.ndarray):
    """Per-channel mean/std over the training samples and buses."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    return mean, np.maximum(std, 1e-6)


def iterate_batches(tensors: dict, batch_size: int, shuffle: bool, generator=None):
    n = len(tensors["X"])
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {k: (v[idx] if getattr(v, "shape", (0,))[:1] == (n,) else v) for k, v in tensors.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def masked_mse(pred, true, pmax):
    """MSE restricted to committed buses.

    Uncommitted buses are excluded rather than trained toward zero: their label
    is zero by definition of the commitment feature the model already sees, so
    including them inflates the apparent fit without teaching any dispatch.
    """
    mask = pmax > 0
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - true[mask]) ** 2).mean()


def overload_penalty(flows, ratings):
    """Hinge penalty on flows exceeding line ratings.

    This is where transmission *capacity* enters the model. The shift operator
    carries reactances only, so without a term like this nothing tells the
    network that a line can bind. Note Prescient's own line limits are soft --
    41 line-hours in the base case exceed their rating, peaking at 148% -- so
    the labels themselves occasionally violate the constraint being penalized.
    """
    excess = torch.relu(flows.abs() - ratings)
    return (excess**2).mean()


def run_epoch(model, tensors, ptdf, device, optimizer=None, batch_size: int = 128, flow_penalty: float = 0.0, balance: bool = True, generator=None):
    """One pass over a split. The loss is taken on the *projected* dispatch.

    Training through `project_dispatch` rather than on the raw network output
    means the network learns to compensate for the box clamp and the balance
    correction instead of fighting them at inference time.
    """
    training = optimizer is not None
    model.train(training)

    total, count = 0.0, 0
    for batch in iterate_batches(tensors, batch_size, shuffle=training, generator=generator):
        X = batch["X"].to(device)
        pmin, pmax = batch["pmin"].to(device), batch["pmax"].to(device)
        fixed = batch["fixed"].to(device)

        pred_mw = project_dispatch(model(X), pmin, pmax, required_total(fixed) if balance else None)
        pred_frac = pred_mw / pmax.clamp_min(1e-6)
        loss = masked_mse(pred_frac, batch["y_frac"].to(device), pmax)

        if flow_penalty > 0:
            flows = branch_flows(pred_mw, fixed, ptdf.to(device), batch["gen_idx"].to(device))
            loss = loss + flow_penalty * overload_penalty(flows, batch["line_ratings"].to(device))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += float(loss.detach()) * len(X)
        count += len(X)

    return total / max(count, 1)


def train_model(model, name, splits, ptdf, device, args):
    """Fit one model, keeping the epoch with the best validation dispatch MAE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name} | {n_params:,} trainable parameters ===")

    best = {"val_mae": float("inf"), "epoch": 0, "state": None}
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, splits["train"], ptdf, device, optimizer,
            batch_size=args.batch_size, flow_penalty=args.flow_penalty,
            balance=not args.no_balance, generator=generator,
        )
        val = evaluate(model, splits["val"], ptdf, device, balance=not args.no_balance)

        if val["dispatch_mae_mw"] < best["val_mae"] - 1e-6:
            best = {"val_mae": val["dispatch_mae_mw"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            patience = 0
        else:
            patience += 1

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"  epoch {epoch:4d} | train loss {train_loss:.5f} | "
                f"val MAE {val['dispatch_mae_mw']:7.3f} MW "
                f"(interior {val['dispatch_mae_mw_interior']:7.3f}) | "
                f"flow MAE {val['flow_mae_mw']:6.2f} MW"
            )

        if args.patience and patience >= args.patience:
            print(f"  early stop at epoch {epoch} (no val improvement for {patience})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, n_params, best["epoch"]


def report(name, metrics):
    print(f"\n--- {name} ---")
    print(f"  dispatch MAE        {metrics['dispatch_mae_mw']:8.3f} MW   (committed buses)")
    print(f"    interior          {metrics['dispatch_mae_mw_interior']:8.3f} MW   <-- the real number")
    print(f"    at a bound        {metrics['dispatch_mae_mw_at_bound']:8.3f} MW   (pinned by inputs)")
    print(f"    congested hours   {metrics['dispatch_mae_mw_congested']:8.3f} MW   "
          f"({metrics['n_hours_congested']} of {metrics['n_hours']} h)")
    print(f"    uncongested hours {metrics['dispatch_mae_mw_uncongested']:8.3f} MW")
    print(f"  system total MAE    {metrics['total_dispatch_mae_mw']:8.3f} MW")
    print(f"  flow MAE            {metrics['flow_mae_mw']:8.3f} MW   (vs Prescient)")
    print(f"  congestion F1       {metrics['congestion_f1']:8.3f}      "
          f"(precision {metrics['congestion_precision']:.3f}, recall {metrics['congestion_recall']:.3f})")
    print(f"  balance residual    {metrics['balance_residual_mw']:8.3f} MW")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(out_dir: Path, model, name: str, model_class: str, model_config: dict, norm_stats: tuple, metrics: dict, extra: dict | None = None):
    """Save a self-contained bundle plus a readable metadata sidecar.

    The checkpoint carries the feature normalization statistics because the
    model is meaningless without them: feeding raw MW-scale features to a
    network trained on z-scored inputs produces confident nonsense rather than
    an error.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = norm_stats
    checkpoint = {
        "format_version": 1,
        "model_class": model_class,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "feature_mean": torch.as_tensor(mean),
        "feature_std": torch.as_tensor(std),
        "metrics": metrics,
        "torch_version": torch.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        checkpoint.update(extra)

    path = out_dir / f"{name}.pt"
    torch.save(checkpoint, path)

    metadata = {
        "model_class": model_class,
        "model_config": {k: v for k, v in model_config.items() if not isinstance(v, np.ndarray)},
        "metrics": metrics,
        "torch_version": torch.__version__,
        "saved_at": checkpoint["saved_at"],
        "checkpoint_file": path.name,
    }
    if extra:
        metadata.update({k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool, list, dict))})
    (out_dir / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_model(path: Path, shift: np.ndarray | None = None, gen_node_idx: np.ndarray | None = None, device=None):
    """Rebuild a trained model from a `save_model` checkpoint.

    OPFGNN's constructor needs the shift operator and generator-node indices,
    which are dataset artifacts rather than weights; pass them in, or let them
    be read from the default dataset.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = dict(checkpoint["model_config"])

    if checkpoint["model_class"] == "OPFGNN":
        if shift is None or gen_node_idx is None:
            data = data_prep.load_dataset()
            shift, gen_node_idx = data["S"], data["gen_node_idx"]
        model = OPFGNN(shift=shift, gen_node_idx=gen_node_idx, **config)
    else:
        model = OPFMLP(**config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a GNN surrogate for PCM DC economic dispatch.")
    parser.add_argument("--dataset", type=str, default=str(data_prep.DATASET_DIR / data_prep.DEFAULT_DATASET_NAME))
    parser.add_argument("--model", choices=["gnn", "mlp", "both"], default="both",
                        help="'both' (default) trains the FCNN baseline alongside the GNN.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 16], help="GNN filter widths per layer.")
    parser.add_argument("--taps", type=int, default=3, help="Filter taps K (hop reach) per graph-filter layer.")
    parser.add_argument("--readout", choices=["local", "global"], default="local")
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=40, help="Early-stopping patience in epochs; 0 disables.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-penalty", type=float, default=0.0,
                        help="Weight on the PTDF line-overload hinge penalty. 0 disables it.")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip the power-balance projection, leaving the per-bus predictions to sum to whatever they sum to. "
                             "Useful only for measuring what the projection is worth.")
    parser.add_argument("--out-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.dataset)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run `python data_prep.py` first.")
    data = data_prep.load_dataset(data_path)

    v = data["ptdf_validation"]
    print(f"dataset            : {data_path}")
    print(f"X                  : {tuple(data['X'].shape)}  features {list(data['feature_names'])}")
    print(f"y                  : {tuple(data['y'].shape)}  buses {list(data['gen_bus_ids'])}")
    print(f"PTDF check vs PCM  : MAE {v['mae_mw']:.4f} MW (corr {v['corr']:.6f})")

    masks = {k: data[f"{k}_mask"] for k in ("train", "val", "test")}
    mean, std = feature_stats(data["X"][masks["train"]])
    splits = {k: make_split_tensors(data, m, mean, std) for k, m in masks.items()}
    print(
        f"split              : train {len(splits['train']['X'])} / "
        f"val {len(splits['val']['X'])} / test {len(splits['test']['X'])}  (contiguous month blocks)"
    )

    ptdf = torch.as_tensor(data["ptdf"], dtype=torch.float32)
    n_nodes, in_features = data["X"].shape[1], data["X"].shape[2]
    n_out = data["y"].shape[1]

    results, saved = {}, {}
    if args.model in ("gnn", "both"):
        config = {
            "in_features": in_features,
            "hidden_features": tuple(args.hidden),
            "taps": args.taps,
            "readout": args.readout,
            "dropout": args.dropout,
        }
        model = OPFGNN(shift=data["S"], gen_node_idx=data["gen_node_idx"], **config).to(device)
        model, n_params, best_epoch = train_model(model, "OPFGNN", splits, ptdf, device, args)
        results["OPFGNN"] = evaluate(model, splits["test"], ptdf, device, balance=not args.no_balance)
        results["OPFGNN"].update({"n_parameters": n_params, "best_epoch": best_epoch})
        saved["OPFGNN"] = (model, "OPFGNN", config)

    if args.model in ("mlp", "both"):
        config = {
            "n_nodes": n_nodes,
            "in_features": in_features,
            "n_out": n_out,
            "hidden": tuple(args.mlp_hidden),
            "dropout": args.dropout,
        }
        model = OPFMLP(**config).to(device)
        model, n_params, best_epoch = train_model(model, "OPFMLP (baseline)", splits, ptdf, device, args)
        results["OPFMLP"] = evaluate(model, splits["test"], ptdf, device, balance=not args.no_balance)
        results["OPFMLP"].update({"n_parameters": n_params, "best_epoch": best_epoch})
        saved["OPFMLP"] = (model, "OPFMLP", config)

    print("\n" + "=" * 68)
    print("TEST RESULTS (Nov-Dec hold-out)")
    print("=" * 68)
    for name, metrics in results.items():
        report(f"{name}  ({metrics['n_parameters']:,} params, best epoch {metrics['best_epoch']})", metrics)

    if len(results) == 2:
        gnn, mlp = results["OPFGNN"], results["OPFMLP"]
        delta = mlp["dispatch_mae_mw_interior"] - gnn["dispatch_mae_mw_interior"]
        print(
            f"\ninterior MAE, MLP - GNN = {delta:+.3f} MW with "
            f"{mlp['n_parameters'] / max(gnn['n_parameters'], 1):.1f}x the parameters."
        )
        if delta <= 0:
            print("The graph structure is not earning its place here -- consistent with "
                  "Falconer & Mones on fixed topology at this system size.")

    for name, (model, model_class, config) in saved.items():
        path = save_model(
            out_dir=args.out_dir,
            model=model,
            name=name.lower(),
            model_class=model_class,
            model_config=config,
            norm_stats=(mean, std),
            metrics=results[name],
            extra={
                "feature_names": list(map(str, data["feature_names"])),
                "gen_bus_ids": data["gen_bus_ids"].tolist(),
                "pcm_runs": list(map(str, data["pcm_runs"])),
                "dataset": str(data_path),
                "training_args": vars(args),
            },
        )
        print(f"saved {name} -> {path}")


if __name__ == "__main__":
    main()
