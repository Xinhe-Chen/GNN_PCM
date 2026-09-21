"""
Data preparation for the GNN DC-OPF / economic-dispatch surrogate.

Builds a supervised dataset in which one sample is one simulated hour of a
Prescient production-cost-model (PCM) run, laid out on the RTS-GMLC bus graph:

    X [T, N, F]   node features, N = 73 buses
    y [T, M]      labels: bus-aggregate thermal dispatch, M = 31 thermal buses
    S [N, N]      graph shift operator (normalized susceptance matrix)
    PTDF [L, N]   power transfer distribution factors, L = 120 AC branches

Why these features and not the AC-OPF ones
------------------------------------------
The reference formulation for GNN-based OPF (Owerko, Gama & Ribeiro, ICASSP
2020) uses a four-channel node state [v, delta, p, q]. Under the DC
approximation that Prescient actually solves, three of those four carry no
information:

    v     == 1.0 everywhere      (constant)
    q     == 0                   (not modeled)
    delta == B^+ p               (a *linear* function of the injections, which
                                  a single graph-convolution layer with S = B
                                  reproduces exactly and for free)

So the signal has to come from somewhere else. What actually varies hour to
hour in a PCM run is which units are committed, how much capacity that leaves,
and what it costs -- hence channels 2-5 below. Dropping commitment would make
the label unrecoverable: a bus with two cheap steam units online behaves
nothing like the same bus with one peaking CT online.

Feature channels (F = 8)
------------------------
    0  load_mw                  bus active demand                      [time-varying]
    1  renewable_avail_mw       available (pre-curtailment) renewables  [time-varying]
    2  committed_pmax_mw        sum of PMax over committed units        [time-varying]
    3  committed_pmin_mw        sum of PMin over committed units        [time-varying]
    4  committed_mc             capacity-weighted marginal cost $/MWh   [time-varying]
    5  committed_unit_count     number of committed units               [time-varying]
    6  incident_line_cap_mw     sum of Cont Rating over incident lines  [static]
    7  incident_cap_per_susc    incident rating / incident susceptance  [static]

Channels 0-5 are per-bus, per-hour. Channels 6-7 are static topology
attributes, broadcast over time: they are a deliberately crude stand-in for
transmission *ratings*, which the shift operator S does not carry. S is built
from reactances, so it encodes how flow distributes but says nothing about when
a line binds -- and congestion is the only thing that makes DC-OPF more than a
merit-order sort. In this PCM run congestion is present in ~19% of hours but
absent in the rest, so a model with no notion of line capacity will score well
on aggregate while being structurally blind to the regime that matters. The
principled fix is per-edge features with edge-conditioned message passing;
channels 6-7 buy some of that signal without leaving the polynomial-filter
architecture.

Channels deliberately EXCLUDED as label leakage
-----------------------------------------------
    thermal_detail "Headroom"    == PMax - Dispatch, i.e. the label, negated
    thermal_detail "Unit Cost"   a function of the realized dispatch
    renewables_detail "Output"   a PCM dispatch decision, not an input
                                 (channel 1 uses *available* renewables, which
                                 is exogenous; Output is used only to build
                                 `net_injection_fixed`, see below)

Labels
------
y is Prescient's `thermal_detail.Dispatch` summed to the bus. It is used
directly rather than re-solved: a check on this run found ramp constraints
never binding (zero of 159,307 on->on transitions within 5% of the hourly ramp
limit; mean hour-to-hour change 4.4% of the limit), so given commitment the
dispatch really is a function of the snapshot. Re-solving a single-period
DC-OPF instead would *introduce* error, because Prescient also holds ~416
MW/hour of reserves across six products that a snapshot OPF would not.

Flow reconstruction
-------------------
Under DC assumptions branch flows are exact in the injections: f = PTDF @ p.
Validated against this run's `line_detail.csv` to 0.094 MW MAE over all
8,784 hours x 120 lines (correlation 0.999999) -- see `validate_ptdf`. Two
details matter for that agreement:

  * DC1 is a 100 MW DC tie (bus 113 -> 316) that obeys no PTDF. It must be
    applied as an injection pair, not as a branch. Ignoring it costs 8.9 MW
    MAE instead of 0.09 MW.
  * Transformer tap ratios can be ignored; plain 1/X is enough.

`net_injection_fixed` [T, N] holds everything in the injection vector that is
NOT thermal dispatch (renewable output - demand, plus the DC1 pair). The
trainer adds its own predicted dispatch and multiplies by PTDF. It is a
diagnostic / physics-loss input only -- never a node feature, since it embeds
the PCM's curtailment decisions.

Source files (relative to repo root)
------------------------------------
    data/GMLC_source_data/bus.csv, branch.csv, gen.csv
    data/GMLC_source_data/bus_load.csv         [T, N] hourly bus demand
    data/GMLC_source_data/bus_renewable.csv    [T, N] hourly available renewables
    data/PCM_results/<run>/thermal_detail.csv
    data/PCM_results/<run>/renewables_detail.csv
    data/PCM_results/<run>/line_detail.csv
    data/PCM_results/<run>/hourly_summary.csv
    data/PCM_results/<run>/bus_detail.csv      (LMP, for the congestion flag)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "GMLC_source_data"
BUS_CSV = DATA_DIR / "bus.csv"
BRANCH_CSV = DATA_DIR / "branch.csv"
GEN_CSV = DATA_DIR / "gen.csv"
BUS_LOAD_CSV = DATA_DIR / "bus_load.csv"
BUS_RENEWABLE_CSV = DATA_DIR / "bus_renewable.csv"

PCM_RESULTS_DIR = REPO_ROOT / "data" / "PCM_results"
DEFAULT_PCM_RUN = "base_case_pcm_test"

DATASET_DIR = REPO_ROOT / "data" / "model"
DEFAULT_DATASET_NAME = "opf_dcopf_dataset.npz"

#: Node feature channel names, in column order. Channels 6-7 are static.
FEATURE_NAMES = (
    "load_mw",
    "renewable_avail_mw",
    "committed_pmax_mw",
    "committed_pmin_mw",
    "committed_mc",
    "committed_unit_count",
    "incident_line_cap_mw",
    "incident_cap_per_susc",
)
N_STATIC_FEATURES = 2

#: Unit types treated as dispatchable thermal generation, i.e. the units whose
#: output the model predicts. SYNC_COND is excluded: synchronous condensers
#: appear in thermal_detail with Dispatch == 0 always (they supply reactive
#: power only), so including them would pad the label with constant zeros.
THERMAL_UNIT_TYPES = ("CC", "CT", "NUCLEAR", "STEAM")

#: Flow on the DC tie is at its +/-100 MW rating in almost every hour, so the
#: schedule can either be read from the PCM output or pinned at the rating.
DC_BRANCH_UID = "DC1"


# ---------------------------------------------------------------------------
# Topology, shift operator, PTDF
# ---------------------------------------------------------------------------

def load_topology(bus_csv: Path = BUS_CSV, branch_csv: Path = BRANCH_CSV):
    """Build the fixed bus-level graph from bus.csv and branch.csv.

    Returns
    -------
    bus_ids : np.ndarray [N]      Bus IDs in node order.
    bus_index : dict[int, int]    Bus ID -> node index 0..N-1.
    edge_index : np.ndarray [2, E]
    adjacency : np.ndarray [N, N] symmetric 0/1
    """
    bus_df = pd.read_csv(bus_csv)
    branch_df = pd.read_csv(branch_csv)

    bus_ids = bus_df["Bus ID"].to_numpy()
    bus_index = {int(b): i for i, b in enumerate(bus_ids)}

    from_idx = branch_df["From Bus"].map(bus_index).to_numpy()
    to_idx = branch_df["To Bus"].map(bus_index).to_numpy()
    edge_index = np.stack([from_idx, to_idx], axis=0)

    n = len(bus_ids)
    adjacency = np.zeros((n, n), dtype=int)
    adjacency[from_idx, to_idx] = 1
    adjacency[to_idx, from_idx] = 1

    return bus_ids, bus_index, edge_index, adjacency


def build_susceptance_matrices(bus_index: dict, branch_csv: Path = BRANCH_CSV):
    """Assemble the DC power-flow matrices B_bus [N, N] and B_f [L, N].

    B_bus is the standard bus susceptance matrix (b_ij = 1/x_ij accumulated on
    the diagonal, negated off-diagonal); B_f maps bus angles to branch flows.
    Parallel branches accumulate in B_bus automatically but stay separate rows
    in B_f, since each physical line has its own rating.

    Transformer tap ratios ("Tr Ratio", non-unity on 15 RTS-GMLC branches) are
    ignored: plain 1/X reproduces Prescient's flows to 0.09 MW, so the tap
    correction is below the noise floor of the DC approximation itself.
    """
    branch_df = pd.read_csv(branch_csv)
    n, ell = len(bus_index), len(branch_df)

    b_bus = np.zeros((n, n), dtype=np.float64)
    b_f = np.zeros((ell, n), dtype=np.float64)

    for k, row in branch_df.iterrows():
        i, j = bus_index[int(row["From Bus"])], bus_index[int(row["To Bus"])]
        b = 1.0 / float(row["X"])
        b_f[k, i] += b
        b_f[k, j] -= b
        b_bus[i, i] += b
        b_bus[j, j] += b
        b_bus[i, j] -= b
        b_bus[j, i] -= b

    # Cast to a fixed-width unicode dtype: pandas hands back object-dtype
    # string arrays, which np.savez cannot store without allow_pickle.
    line_ids = branch_df["UID"].to_numpy().astype(np.str_)
    line_ratings = branch_df["Cont Rating"].to_numpy(dtype=np.float64)
    return b_bus, b_f, line_ids, line_ratings


def build_shift_operator(b_bus: np.ndarray):
    """Normalize the susceptance matrix into a GNN shift operator S.

    Dividing by the largest eigenvalue magnitude keeps the polynomial filters
    Sum_k H_k S^k numerically stable: without it, S^3 on this system has
    entries of order 1e4 and the deeper filter taps dominate regardless of
    their learned weights.

    Using B itself (rather than the Gaussian impedance kernel exp(-k|z|^2) of
    the reference paper) is the physically exact choice here, because DC power
    flow *is* delta = B^+ p. Liu, Wu & Zhu (IEEE TPWRS 2022) likewise
    initialize their graph filters from the normalized B-bus matrix.
    """
    eigenvalues = np.linalg.eigvalsh(b_bus)
    lambda_max = float(np.max(np.abs(eigenvalues)))
    return (b_bus / lambda_max).astype(np.float64), lambda_max


def build_ptdf(b_bus: np.ndarray, b_f: np.ndarray, ref_node: int):
    """Compute the PTDF matrix, Phi = B_f B_bus^-1 with the reference removed.

    B_bus is singular (its null space is the all-ones angle shift), so the
    reference bus row/column is deleted before inversion and the reference
    column of the result is filled with zeros -- injections at the reference
    bus by definition produce no flow in this formulation.

    Returns
    -------
    ptdf : np.ndarray [L, N]
    """
    n = b_bus.shape[0]
    keep = [i for i in range(n) if i != ref_node]
    ptdf = np.zeros((b_f.shape[0], n), dtype=np.float64)
    ptdf[:, keep] = b_f[:, keep] @ np.linalg.inv(b_bus[np.ix_(keep, keep)])
    return ptdf


def reference_node(bus_csv: Path = BUS_CSV, bus_index: dict | None = None):
    """Find the slack/reference bus node index from bus.csv's "Bus Type"."""
    bus_df = pd.read_csv(bus_csv)
    ref_ids = bus_df.loc[bus_df["Bus Type"] == "Ref", "Bus ID"].tolist()
    if len(ref_ids) != 1:
        raise ValueError(f"Expected exactly one Ref bus in {bus_csv.name}, found {ref_ids}")
    ref_id = int(ref_ids[0])
    return (bus_index[ref_id] if bus_index else ref_id), ref_id


def incident_line_features(bus_index: dict, branch_csv: Path = BRANCH_CSV):
    """Static per-bus summaries of the transmission capacity around each bus.

    Returns two [N] arrays:
      * total Cont Rating of the lines incident to the bus, an upper bound on
        how much power the bus can export or import at all;
      * that rating divided by the total incident susceptance, a rough measure
        of how *constrained* the bus's neighbourhood is relative to how
        strongly it is electrically coupled.

    Both are constant over time, so for a fixed topology they act as node
    position encodings rather than as true edge features. They exist because
    the shift operator carries reactances but not ratings; see the module
    docstring.
    """
    branch_df = pd.read_csv(branch_csv)
    n = len(bus_index)
    cap = np.zeros(n, dtype=np.float64)
    susc = np.zeros(n, dtype=np.float64)

    for _, row in branch_df.iterrows():
        i, j = bus_index[int(row["From Bus"])], bus_index[int(row["To Bus"])]
        rating = float(row["Cont Rating"])
        b = 1.0 / float(row["X"])
        for node in (i, j):
            cap[node] += rating
            susc[node] += b

    ratio = np.divide(cap, susc, out=np.zeros_like(cap), where=susc > 0)
    return cap, ratio


# ---------------------------------------------------------------------------
# Time series and PCM results
# ---------------------------------------------------------------------------

def load_bus_timeseries(bus_ids: np.ndarray, csv_path: Path):
    """Read a [T, N] hourly per-bus CSV (bus_load.csv / bus_renewable.csv).

    Both files are indexed by hour-of-year with one column per Bus ID. Columns
    are reordered onto `bus_ids` so that column n means the same bus in every
    array this module returns; buses absent from the file (no load, or no
    renewable resource) become all-zero columns rather than an error.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.astype(int)
    df = df.reindex(columns=[int(b) for b in bus_ids], fill_value=0.0)
    return df.to_numpy(dtype=np.float64)


def thermal_unit_table(gen_csv: Path = GEN_CSV):
    """Per-unit thermal attributes needed for the commitment-derived features.

    `marginal_cost` is the average-heat-rate cost at full output:
    Fuel Price [$/MMBTU] * HR_avg_0 [BTU/kWh] / 1000 + VOM [$/MWh]. RTS-GMLC
    actually specifies piecewise-linear incremental heat rates (HR_incr_1..4);
    the average rate is used because the feature only needs to place the bus in
    the merit order, not to reproduce the cost curve.

    Returns a DataFrame indexed by GEN UID with columns
    [bus_id, unit_type, pmax_mw, pmin_mw, marginal_cost], restricted to
    THERMAL_UNIT_TYPES.
    """
    gen_df = pd.read_csv(gen_csv)
    gen_df = gen_df[gen_df["Unit Type"].isin(THERMAL_UNIT_TYPES)].copy()

    def col(name):
        return pd.to_numeric(gen_df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    out = pd.DataFrame(
        {
            "bus_id": gen_df["Bus ID"].to_numpy(dtype=int),
            "unit_type": gen_df["Unit Type"].to_numpy(),
            "pmax_mw": col("PMax MW"),
            "pmin_mw": col("PMin MW"),
            "marginal_cost": col("Fuel Price $/MMBTU") * col("HR_avg_0") / 1000.0 + col("VOM"),
        },
        index=gen_df["GEN UID"].to_numpy(),
    )
    out.index.name = "GEN UID"
    return out


def _hour_key(df: pd.DataFrame):
    """Add a dense 0..T-1 hour index from the (Date, Hour) columns.

    Prescient writes long-format detail files with a Minute column that is
    always 0 in an hourly run. Grouping on (Date, Hour) and sorting gives a
    stable chronological index shared by every detail file, which is what lets
    thermal/renewable/line/bus tables be aligned by position afterwards.
    """
    keys = df[["Date", "Hour"]].drop_duplicates().sort_values(["Date", "Hour"]).reset_index(drop=True)
    keys["_t"] = np.arange(len(keys))
    return df.merge(keys, on=["Date", "Hour"], how="left"), keys


def load_commitment_and_dispatch(run_dir: Path, units: pd.DataFrame, bus_ids: np.ndarray):
    """Aggregate thermal_detail.csv to per-bus, per-hour commitment and dispatch.

    Returns a dict of [T, N] arrays (`committed_pmax`, `committed_pmin`,
    `committed_mc`, `unit_count`, `dispatch`) plus the hour key table and the
    per-unit at-bound counts used for stratified evaluation.

    `committed_mc` is capacity-weighted over the committed units at the bus and
    is 0 where nothing is committed, matching the other channels' convention so
    that an uncommitted bus is an all-zero generation signature rather than a
    bus with a misleading cost.
    """
    df = pd.read_csv(run_dir / "thermal_detail.csv")
    df = df[df["Generator"].isin(units.index)].copy()
    if df.empty:
        raise ValueError(f"No {THERMAL_UNIT_TYPES} units found in {run_dir / 'thermal_detail.csv'}")

    df, keys = _hour_key(df)
    df["Unit State"] = df["Unit State"].astype(bool)
    df["bus_id"] = df["Generator"].map(units["bus_id"])
    df["pmax"] = df["Generator"].map(units["pmax_mw"])
    df["pmin"] = df["Generator"].map(units["pmin_mw"])
    df["mc"] = df["Generator"].map(units["marginal_cost"])

    on = df["Unit State"].to_numpy()
    df["c_pmax"] = np.where(on, df["pmax"], 0.0)
    df["c_pmin"] = np.where(on, df["pmin"], 0.0)
    df["c_mc_w"] = np.where(on, df["mc"] * df["pmax"], 0.0)
    df["c_n"] = on.astype(np.float64)

    n_hours, columns = len(keys), [int(b) for b in bus_ids]

    def to_grid(value_col):
        grid = df.pivot_table(index="_t", columns="bus_id", values=value_col, aggfunc="sum", fill_value=0.0)
        grid = grid.reindex(index=range(n_hours), fill_value=0.0)
        grid = grid.reindex(columns=columns, fill_value=0.0)
        return grid.to_numpy(dtype=np.float64)

    committed_pmax = to_grid("c_pmax")
    mc_weighted = to_grid("c_mc_w")
    committed_mc = np.divide(
        mc_weighted, committed_pmax, out=np.zeros_like(mc_weighted), where=committed_pmax > 0
    )

    # At-bound classification, per unit-hour, then aggregated to the bus. Over
    # half of all committed unit-hours sit exactly at PMin or PMax, where the
    # value is pinned by commitment and capacity rather than by the dispatch
    # decision. Reporting one pooled error would be dominated by those.
    tol = 0.5
    committed = df[df["Unit State"]]
    at_bound = (
        (committed["Dispatch"] >= committed["pmax"] - tol) | (committed["Dispatch"] <= committed["pmin"] + tol)
    )
    frac_at_bound = float(at_bound.mean()) if len(committed) else 0.0

    return {
        "committed_pmax": committed_pmax,
        "committed_pmin": to_grid("c_pmin"),
        "committed_mc": committed_mc,
        "unit_count": to_grid("c_n"),
        "dispatch": to_grid("Dispatch"),
        "keys": keys,
        "frac_unit_hours_at_bound": frac_at_bound,
    }


def load_renewable_output(run_dir: Path, bus_ids: np.ndarray, n_hours: int):
    """Per-bus, per-hour *realized* renewable output [T, N] from the PCM.

    This is a dispatch decision (it embeds curtailment), so it is NOT a node
    feature -- it is only used to build the fixed part of the injection vector
    for flow reconstruction. The exogenous *available* profile in
    bus_renewable.csv is what feeds channel 1.
    """
    df = pd.read_csv(run_dir / "renewables_detail.csv")
    df, _ = _hour_key(df)
    df["bus_id"] = df["Generator"].str.split("_").str[0].astype(int)
    grid = df.pivot_table(index="_t", columns="bus_id", values="Output", aggfunc="sum", fill_value=0.0)
    grid = grid.reindex(index=range(n_hours), fill_value=0.0)
    grid = grid.reindex(columns=[int(b) for b in bus_ids], fill_value=0.0)
    return grid.to_numpy(dtype=np.float64)


def load_line_flows(run_dir: Path, line_ids: np.ndarray, n_hours: int):
    """Prescient's AC-branch flows [T, L] plus the DC tie schedule [T].

    The DC tie is separated out because it carries no PTDF relationship: it is
    a controlled transfer, handled as an injection pair in
    `build_net_injection_fixed`.
    """
    df = pd.read_csv(run_dir / "line_detail.csv")
    df, _ = _hour_key(df)

    ac = df[df["Line"] != DC_BRANCH_UID]
    grid = ac.pivot_table(index="_t", columns="Line", values="Flow", aggfunc="sum")
    grid = grid.reindex(index=range(n_hours)).reindex(columns=list(line_ids))
    flows = grid.to_numpy(dtype=np.float64)

    dc = df[df["Line"] == DC_BRANCH_UID]
    dc_flow = (
        dc.set_index("_t")["Flow"].reindex(range(n_hours)).fillna(0.0).to_numpy(dtype=np.float64)
        if len(dc)
        else np.zeros(n_hours)
    )
    return flows, dc_flow


def build_net_injection_fixed(
    load_mw: np.ndarray,
    renewable_output: np.ndarray,
    dc_flow: np.ndarray,
    bus_index: dict,
    branch_csv: Path = BRANCH_CSV,
    dc_csv: Path | None = None,
):
    """Everything in the injection vector except thermal dispatch, as [T, N].

    net_injection_fixed = renewable_output - load + dc_pair

    The DC1 term injects -flow at its from-bus and +flow at its to-bus, which
    is what makes PTDF reconstruction exact (0.09 MW MAE instead of 8.9 MW).
    The endpoints come from dc_branch.csv when available and fall back to the
    documented RTS-GMLC tie (113 -> 316) otherwise, since dc_branch.csv is not
    part of the preprocessed source-data folder.
    """
    fixed = renewable_output - load_mw

    from_bus, to_bus = 113, 316
    if dc_csv is not None and Path(dc_csv).is_file():
        dc_df = pd.read_csv(dc_csv)
        row = dc_df[dc_df["UID"] == DC_BRANCH_UID]
        if len(row):
            from_bus, to_bus = int(row.iloc[0]["From Bus"]), int(row.iloc[0]["To Bus"])

    if from_bus in bus_index and to_bus in bus_index:
        fixed[:, bus_index[from_bus]] -= dc_flow
        fixed[:, bus_index[to_bus]] += dc_flow

    return fixed


def validate_ptdf(ptdf: np.ndarray, net_injection_fixed: np.ndarray, dispatch: np.ndarray, flows_ref: np.ndarray):
    """Check f = PTDF @ p against the PCM's own reported flows.

    Run this before trusting anything downstream: if the reconstruction does
    not match, the network model (reactances, reference bus, DC tie handling)
    is wrong and no amount of training will fix it. On the base case this
    returns MAE ~0.09 MW against flows of order hundreds of MW.

    The small residual that remains is the two RTS-GMLC units (ROR, STORAGE)
    that Prescient does not write into either detail file, which leaves the
    hourly generation-minus-load balance short by ~5 MW on average.
    """
    injection = net_injection_fixed + dispatch
    predicted = injection @ ptdf.T
    mask = np.isfinite(flows_ref)
    err = predicted[mask] - flows_ref[mask]
    return {
        "mae_mw": float(np.abs(err).mean()),
        "rmse_mw": float(np.sqrt((err**2).mean())),
        "max_abs_err_mw": float(np.abs(err).max()),
        "corr": float(np.corrcoef(predicted[mask], flows_ref[mask])[0, 1]),
        "mean_abs_flow_mw": float(np.abs(flows_ref[mask]).mean()),
    }


# ---------------------------------------------------------------------------
# Sample flags and splits
# ---------------------------------------------------------------------------

def load_sample_flags(run_dir: Path, keys: pd.DataFrame):
    """Per-hour diagnostic flags used for filtering and stratified reporting.

    Returns [T] arrays:
      load_shed_mw        hours where the PCM itself could not serve load; the
                          dispatch there is degenerate (everything at max) and
                          these are dropped by default
      reserve_shortfall   kept, but flagged: reserves do not constrain a
                          snapshot dispatch, so these hours are still valid
                          samples
      lmp_spread          max - min LMP across buses, the congestion
                          indicator. ~88% of hours are exactly 0 (no binding
                          line, so the network is irrelevant and DC-OPF is a
                          merit-order sort); in the rest the spread averages
                          ~$42/MWh, larger than the mean LMP itself. Any
                          unstratified metric is dominated by the easy 88%.
    """
    n_hours = len(keys)

    hourly = pd.read_csv(run_dir / "hourly_summary.csv")
    hourly = keys.merge(hourly, on=["Date", "Hour"], how="left")
    load_shed = hourly["LoadShedding"].fillna(0.0).to_numpy(dtype=np.float64)
    reserve_short = hourly["ReserveShortfall"].fillna(0.0).to_numpy(dtype=np.float64)

    bus_detail = pd.read_csv(run_dir / "bus_detail.csv", usecols=["Date", "Hour", "LMP"])
    spread = bus_detail.groupby(["Date", "Hour"])["LMP"].agg(lambda s: s.max() - s.min()).rename("spread")
    spread = keys.merge(spread.reset_index(), on=["Date", "Hour"], how="left")
    lmp_spread = spread["spread"].fillna(0.0).to_numpy(dtype=np.float64)

    dates = pd.to_datetime(keys["Date"])
    return {
        "load_shed_mw": load_shed[:n_hours],
        "reserve_shortfall_mw": reserve_short[:n_hours],
        "lmp_spread": lmp_spread[:n_hours],
        "month": dates.dt.month.to_numpy(dtype=np.int64),
        "day_of_year": dates.dt.dayofyear.to_numpy(dtype=np.int64),
        "hour_of_day": keys["Hour"].to_numpy(dtype=np.int64),
    }


def month_block_split(month: np.ndarray, train_months=range(1, 10), val_months=(10,), test_months=(11, 12)):
    """Split samples into contiguous month blocks.

    A random split would leak: consecutive hours in a PCM trajectory are
    strongly autocorrelated, so neighbouring hours landing on opposite sides of
    the split makes the test set nearly a copy of the training set. Contiguous
    blocks also give a seasonal-generalization test, which is the more
    defensible claim for a single-trajectory dataset.
    """
    train = np.isin(month, list(train_months))
    val = np.isin(month, list(val_months))
    test = np.isin(month, list(test_months))

    overlap = (train & val) | (train & test) | (val & test)
    if overlap.any():
        raise ValueError("train/val/test month sets overlap")
    return train, val, test


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def resolve_pcm_runs(pcm_runs=None, pcm_results_dir: Path = PCM_RESULTS_DIR):
    """Normalize the `pcm_runs` argument into a list of run directories.

    Accepts None (the default base case), "all", a single name/path, or a
    sequence of them. A run must carry every detail file this module reads, so
    partial result folders are rejected up front rather than failing midway.
    """
    pcm_results_dir = Path(pcm_results_dir)
    required = ("thermal_detail.csv", "renewables_detail.csv", "line_detail.csv", "hourly_summary.csv", "bus_detail.csv")

    def complete(path: Path):
        return all((path / f).is_file() for f in required)

    if pcm_runs is None:
        candidates = [pcm_results_dir / DEFAULT_PCM_RUN]
    elif isinstance(pcm_runs, str) and pcm_runs == "all":
        candidates = [p for p in sorted(pcm_results_dir.iterdir()) if p.is_dir() and complete(p)]
        if not candidates:
            raise FileNotFoundError(f"No complete PCM runs found under {pcm_results_dir}")
    elif isinstance(pcm_runs, (str, Path)):
        candidates = [Path(pcm_runs)]
    elif isinstance(pcm_runs, Sequence):
        candidates = [Path(r) for r in pcm_runs]
    else:
        raise TypeError(f"Unsupported `pcm_runs` value: {pcm_runs!r}")

    resolved = []
    for c in candidates:
        run_dir = c if c.is_absolute() or c.exists() else pcm_results_dir / c
        if not complete(run_dir):
            missing = [f for f in required if not (run_dir / f).is_file()]
            raise FileNotFoundError(f"PCM run '{run_dir}' is missing {missing}")
        resolved.append(run_dir)
    return resolved


def build_dataset(
    pcm_runs=None,
    drop_load_shed_hours: bool = True,
    dc1_from: str = "line_detail",
):
    """Assemble the full DC-OPF surrogate dataset.

    Parameters
    ----------
    pcm_runs
        Which PCM result folders supply labels; see `resolve_pcm_runs`.
        Multiple runs are concatenated along the sample axis.
    drop_load_shed_hours
        Drop hours where the PCM shed load (159 of 8,784 in the base case).
        Their dispatch is degenerate -- essentially everything at maximum --
        so they teach the model a mapping that does not generalize.
    dc1_from
        "line_detail" reads the DC tie's realized schedule from the PCM output;
        "rating" pins it at its nameplate 100 MW from bus 113 to 316. Use
        "rating" when applying the model to a scenario with no PCM output to
        read from -- the tie runs at ~99.4 MW on average, so the approximation
        is tight.

    Returns
    -------
    dict with keys:
        X [T, N, F], feature_names
        y [T, M] dispatch in MW, y_frac [T, M] as a fraction of committed
          capacity, committed_pmax [T, M], committed_pmin [T, M]
        gen_bus_mask [N] bool, gen_bus_ids [M], gen_node_idx [M]
        S [N, N], ptdf [L, N], line_ids [L], line_ratings [L]
        net_injection_fixed [T, N], flows_ref [T, L]
        train_mask / val_mask / test_mask [T] bool
        flags: load_shed_mw, reserve_shortfall_mw, lmp_spread, month, ...
        ptdf_validation: dict of reconstruction errors
    """
    if dc1_from not in ("line_detail", "rating"):
        raise ValueError(f"dc1_from must be 'line_detail' or 'rating', got {dc1_from!r}")

    bus_ids, bus_index, edge_index, adjacency = load_topology()
    b_bus, b_f, line_ids, line_ratings = build_susceptance_matrices(bus_index)
    shift, lambda_max = build_shift_operator(b_bus)
    ref_node, ref_bus_id = reference_node(bus_index=bus_index)
    ptdf = build_ptdf(b_bus, b_f, ref_node)
    incident_cap, incident_ratio = incident_line_features(bus_index)

    units = thermal_unit_table()
    load_all = load_bus_timeseries(bus_ids, BUS_LOAD_CSV)
    renew_avail_all = load_bus_timeseries(bus_ids, BUS_RENEWABLE_CSV)

    run_dirs = resolve_pcm_runs(pcm_runs)
    blocks = []

    for run_dir in run_dirs:
        agg = load_commitment_and_dispatch(run_dir, units, bus_ids)
        keys, n_hours = agg["keys"], len(agg["keys"])

        if n_hours > len(load_all):
            raise ValueError(
                f"Run '{run_dir.name}' has {n_hours} hours but bus_load.csv only has {len(load_all)}; "
                "the preprocessed time series and the PCM run must cover the same horizon."
            )
        load_mw = load_all[:n_hours]
        renew_avail = renew_avail_all[:n_hours]

        renew_out = load_renewable_output(run_dir, bus_ids, n_hours)
        flows_ref, dc_flow = load_line_flows(run_dir, line_ids, n_hours)
        if dc1_from == "rating":
            dc_flow = np.full(n_hours, 100.0)

        fixed = build_net_injection_fixed(
            load_mw, renew_out, dc_flow, bus_index, dc_csv=DATA_DIR / "dc_branch.csv"
        )
        flags = load_sample_flags(run_dir, keys)

        static = np.broadcast_to(
            np.stack([incident_cap, incident_ratio], axis=1)[None, :, :], (n_hours, len(bus_ids), N_STATIC_FEATURES)
        )
        X = np.concatenate(
            [
                np.stack(
                    [
                        load_mw,
                        renew_avail,
                        agg["committed_pmax"],
                        agg["committed_pmin"],
                        agg["committed_mc"],
                        agg["unit_count"],
                    ],
                    axis=2,
                ),
                static,
            ],
            axis=2,
        ).astype(np.float32)

        blocks.append(
            {
                "X": X,
                "dispatch_all": agg["dispatch"],
                "committed_pmax_all": agg["committed_pmax"],
                "committed_pmin_all": agg["committed_pmin"],
                "net_injection_fixed": fixed,
                "flows_ref": flows_ref,
                "run_name": np.full(n_hours, run_dir.name),
                "frac_unit_hours_at_bound": agg["frac_unit_hours_at_bound"],
                **flags,
            }
        )

    def cat(key):
        return np.concatenate([b[key] for b in blocks], axis=0)

    X = cat("X")
    dispatch_all = cat("dispatch_all")
    committed_pmax_all = cat("committed_pmax_all")
    committed_pmin_all = cat("committed_pmin_all")
    net_injection_fixed = cat("net_injection_fixed")
    flows_ref = cat("flows_ref")

    # Generator buses are defined by the *static* unit table, not by what
    # happens to be committed: a bus that is off all December still has to be a
    # column of y, otherwise the label layout would change between splits.
    thermal_bus_ids = np.array(sorted(set(units["bus_id"].tolist())))
    gen_node_idx = np.array([bus_index[int(b)] for b in thermal_bus_ids])
    gen_bus_mask = np.zeros(len(bus_ids), dtype=bool)
    gen_bus_mask[gen_node_idx] = True

    y = dispatch_all[:, gen_node_idx]
    committed_pmax = committed_pmax_all[:, gen_node_idx]
    committed_pmin = committed_pmin_all[:, gen_node_idx]
    y_frac = np.divide(y, committed_pmax, out=np.zeros_like(y), where=committed_pmax > 0)

    ptdf_validation = validate_ptdf(ptdf, net_injection_fixed, dispatch_all, flows_ref)

    load_shed = cat("load_shed_mw")
    month = cat("month")
    keep = np.ones(len(X), dtype=bool)
    if drop_load_shed_hours:
        keep &= load_shed <= 0

    train_mask, val_mask, test_mask = month_block_split(month)
    dataset = {
        "X": X,
        "feature_names": np.array(FEATURE_NAMES),
        "y": y.astype(np.float32),
        "y_frac": y_frac.astype(np.float32),
        "committed_pmax": committed_pmax.astype(np.float32),
        "committed_pmin": committed_pmin.astype(np.float32),
        "bus_ids": bus_ids,
        "gen_bus_ids": thermal_bus_ids,
        "gen_node_idx": gen_node_idx,
        "gen_bus_mask": gen_bus_mask,
        "edge_index": edge_index,
        "adjacency": adjacency,
        "S": shift.astype(np.float32),
        "lambda_max": np.float64(lambda_max),
        "ptdf": ptdf.astype(np.float32),
        "line_ids": line_ids,
        "line_ratings": line_ratings.astype(np.float32),
        "ref_bus_id": np.int64(ref_bus_id),
        "ref_node": np.int64(ref_node),
        "net_injection_fixed": net_injection_fixed.astype(np.float32),
        "flows_ref": flows_ref.astype(np.float32),
        "keep_mask": keep,
        "train_mask": train_mask & keep,
        "val_mask": val_mask & keep,
        "test_mask": test_mask & keep,
        "load_shed_mw": load_shed.astype(np.float32),
        "reserve_shortfall_mw": cat("reserve_shortfall_mw").astype(np.float32),
        "lmp_spread": cat("lmp_spread").astype(np.float32),
        "month": month,
        "day_of_year": cat("day_of_year"),
        "hour_of_day": cat("hour_of_day"),
        "run_name": cat("run_name"),
        "pcm_runs": np.array([p.name for p in run_dirs]),
        "frac_unit_hours_at_bound": np.float64(blocks[0]["frac_unit_hours_at_bound"]),
    }
    dataset["ptdf_validation"] = ptdf_validation
    return dataset


def save_dataset(dataset: dict, path: Path):
    """Write the dataset to a compressed .npz.

    `ptdf_validation` is a dict, which npz cannot store, so it is flattened
    into `ptdf_validation_<key>` scalars. Everything else round-trips.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = {}
    for k, v in dataset.items():
        if k == "ptdf_validation":
            continue
        arr = np.asarray(v)
        # Object-dtype (pandas string) arrays would force allow_pickle=True on
        # load, which `load_dataset` deliberately refuses.
        flat[k] = arr.astype(np.str_) if arr.dtype == object else arr
    for k, v in dataset.get("ptdf_validation", {}).items():
        flat[f"ptdf_validation_{k}"] = np.float64(v)

    np.savez_compressed(path, **flat)
    return path


def load_dataset(path: Path = DATASET_DIR / DEFAULT_DATASET_NAME):
    """Read a dataset written by `save_dataset` back into a dict."""
    path = Path(path)
    if path.is_dir():
        path = path / DEFAULT_DATASET_NAME
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}

    validation = {k[len("ptdf_validation_") :]: float(v) for k, v in data.items() if k.startswith("ptdf_validation_")}
    data = {k: v for k, v in data.items() if not k.startswith("ptdf_validation_")}
    data["ptdf_validation"] = validation
    return data


def main():
    parser = argparse.ArgumentParser(description="Build the GNN DC-OPF training dataset from PCM results.")
    parser.add_argument(
        "--pcm-runs",
        nargs="+",
        default=[DEFAULT_PCM_RUN],
        help="PCM result folders under data/PCM_results. 'all' merges every complete run.",
    )
    parser.add_argument(
        "--drop-load-shed-hours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude hours where the PCM shed load (degenerate dispatch).",
    )
    parser.add_argument(
        "--dc1-from",
        choices=["line_detail", "rating"],
        default="line_detail",
        help="Source of the DC tie schedule used in the injection vector.",
    )
    parser.add_argument("--out", type=str, default=str(DATASET_DIR / DEFAULT_DATASET_NAME))
    args = parser.parse_args()

    pcm_runs = "all" if args.pcm_runs == ["all"] else args.pcm_runs
    data = build_dataset(
        pcm_runs=pcm_runs,
        drop_load_shed_hours=args.drop_load_shed_hours,
        dc1_from=args.dc1_from,
    )

    n_t, n_n, n_f = data["X"].shape
    print(f"PCM runs          : {list(data['pcm_runs'])}")
    print(f"samples (hours)   : {n_t}  kept {int(data['keep_mask'].sum())}")
    print(f"X                 : {data['X'].shape}  ({n_n} buses x {n_f} features)")
    print(f"features          : {list(data['feature_names'])}")
    print(f"y                 : {data['y'].shape}  ({len(data['gen_bus_ids'])} thermal buses)")
    print(f"S                 : {data['S'].shape}  lambda_max {float(data['lambda_max']):.2f}")
    print(f"PTDF              : {data['ptdf'].shape}  ref bus {int(data['ref_bus_id'])}")
    print(
        f"split             : train {int(data['train_mask'].sum())} / "
        f"val {int(data['val_mask'].sum())} / test {int(data['test_mask'].sum())}"
    )

    v = data["ptdf_validation"]
    print(
        f"\nPTDF check vs PCM : MAE {v['mae_mw']:.4f} MW | RMSE {v['rmse_mw']:.4f} | "
        f"max {v['max_abs_err_mw']:.2f} | corr {v['corr']:.6f}  "
        f"(mean |flow| {v['mean_abs_flow_mw']:.1f} MW)"
    )
    if v["mae_mw"] > 1.0:
        print("  WARNING: reconstruction error is large -- check reactances, reference bus, DC tie handling.")

    congested = data["lmp_spread"] > 1.0
    print(f"congested hours   : {int(congested.sum())} of {n_t} ({100 * congested.mean():.1f}%)")
    print(f"unit-hours at a dispatch bound: {100 * float(data['frac_unit_hours_at_bound']):.1f}%")

    out = save_dataset(data, Path(args.out))
    print(f"\nsaved dataset to {out}")


if __name__ == "__main__":
    main()
