"""
Data preparation for GNN-based unit commitment / SCUC modeling.

Builds, from the GMLC source data:
    1. Base topology: bus list (nodes), line list (edges), adjacency matrix / edge index.
    2. Node features NF [N, T]: 24-hour nodal demand profile for every bus.
    3. Edge features EF [E, 2]: line susceptance and thermal (continuous) rating.
    4. Generator commitment status [G, T]: on/off (1/0) for every generator, every hour.
    5. Optional node features NR [N, T]: hourly available renewable generation per
       bus (enabled with `build_dataset(include_renewable=True)`).

Source files (relative to repo root):
    data/GMLC_source_data/bus.csv
    data/GMLC_source_data/branch.csv
    data/GMLC_source_data/bus_load.csv
    data/GMLC_source_data/bus_renewable.csv
    data/PCM_results/base_case_pcm_test/thermal_detail.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "GMLC_source_data"
BUS_CSV = DATA_DIR / "bus.csv"
BRANCH_CSV = DATA_DIR / "branch.csv"
BUS_LOAD_CSV = DATA_DIR / "bus_load.csv"
BUS_RENEWABLE_CSV = DATA_DIR / "bus_renewable.csv"
THERMAL_DETAIL_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "PCM_results" / "base_case_pcm_test" / "thermal_detail.csv"
)


def load_topology(bus_csv: Path = BUS_CSV, branch_csv: Path = BRANCH_CSV):
    """Build the fixed graph topology from bus.csv and branch.csv.

    Returns
    -------
    bus_ids : np.ndarray [N]
        Bus IDs in node order (row i of all node/edge arrays refers to bus_ids[i]).
    bus_index : dict[int, int]
        Maps a Bus ID to its node index (0..N-1).
    edge_index : np.ndarray [2, E]
        From-node/to-node indices for each line, in branch.csv row order.
    adjacency : np.ndarray [N, N]
        Symmetric 0/1 adjacency matrix built from the from-bus/to-bus pairs.
    """
    bus_df = pd.read_csv(bus_csv)
    branch_df = pd.read_csv(branch_csv)

    bus_ids = bus_df["Bus ID"].to_numpy()
    bus_index = {bus_id: i for i, bus_id in enumerate(bus_ids)}

    from_idx = branch_df["From Bus"].map(bus_index).to_numpy()
    to_idx = branch_df["To Bus"].map(bus_index).to_numpy()
    edge_index = np.stack([from_idx, to_idx], axis=0)

    n = len(bus_ids)
    adjacency = np.zeros((n, n), dtype=int)
    adjacency[from_idx, to_idx] = 1
    adjacency[to_idx, from_idx] = 1

    return bus_ids, bus_index, edge_index, adjacency


def _load_bus_timeseries(bus_ids: np.ndarray, csv_path: Path, hours: int | None = 24, start_hour: int = 0):
    """Read an hourly per-bus time series CSV into a [N, T] / [N, 24, D] array.

    Both bus_load.csv and bus_renewable.csv share this layout: one row per
    hour of the year (index column) and one column per Bus ID. Selects a
    contiguous `hours`-length window starting at `start_hour`, and reorders
    columns to match `bus_ids`. With `hours=None` the entire year is pulled
    and reshaped into 24 hours x D days (mirroring `load_commitment_status`)
    whenever the horizon is a whole number of days.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.astype(int)

    end = None if hours is None else start_hour + hours
    window = df.iloc[start_hour:end]
    series = window[bus_ids].to_numpy().T  # [N, T]

    n_hours = series.shape[1]
    if hours is None and n_hours % 24 == 0:
        n_days = n_hours // 24
        series = series.reshape(len(bus_ids), n_days, 24).transpose(0, 2, 1)

    return series


def load_node_features(
    bus_ids: np.ndarray, bus_load_csv: Path = BUS_LOAD_CSV, hours: int | None = 24, start_hour: int = 0
):
    """Extract the T-hour nodal demand profile for every bus.

    Returns
    -------
    NF : np.ndarray [N, T] or [N, 24, D]
    """
    return _load_bus_timeseries(bus_ids, bus_load_csv, hours=hours, start_hour=start_hour)


def load_renewable_features(
    bus_ids: np.ndarray, bus_renewable_csv: Path = BUS_RENEWABLE_CSV, hours: int | None = 24, start_hour: int = 0
):
    """Extract the T-hour available renewable generation profile for every bus.

    Same layout and windowing as `load_node_features`; buses with no
    renewable resource are simply all-zero rows.

    Returns
    -------
    NR : np.ndarray [N, T] or [N, 24, D]
    """
    return _load_bus_timeseries(bus_ids, bus_renewable_csv, hours=hours, start_hour=start_hour)


def generator_bus_map(generator_ids: np.ndarray, bus_index: dict):
    """Map each generator to its bus node index, from the generator name prefix.

    Generator names follow the "<BusID>_<Type>_<Unit#>" convention (e.g.
    "101_CT_1" belongs to Bus ID 101).

    Returns
    -------
    gen_bus_idx : np.ndarray [G]
        Node index (into `bus_ids`/`bus_index`) of each generator's bus.
    """
    bus_ids_of_gen = np.array([int(name.split("_")[0]) for name in generator_ids])
    gen_bus_idx = np.array([bus_index[b] for b in bus_ids_of_gen])
    return gen_bus_idx


def load_edge_features(branch_csv: Path = BRANCH_CSV):
    """Extract per-line susceptance and thermal (continuous) rating.

    Susceptance is computed from reactance X as B = 1 / X (per-unit line
    series susceptance), which is the standard DC power-flow quantity used
    for GNN edge features. The line's continuous MVA rating ("Cont Rating")
    is used as the thermal limit.

    Returns
    -------
    EF : np.ndarray [E, 2]
        Columns: [susceptance, thermal_limit].
    """
    branch_df = pd.read_csv(branch_csv)

    susceptance = 1.0 / branch_df["X"].to_numpy()
    thermal_limit = branch_df["Cont Rating"].to_numpy()

    EF = np.stack([susceptance, thermal_limit], axis=1)
    return EF


def load_commitment_status(thermal_detail_csv: Path = THERMAL_DETAIL_CSV, hours: int | None = None, start_hour: int = 0):
    """Extract generator commitment status (0/1) for every generator, every hour.

    thermal_detail.csv is in long format: one row per (Date, Hour, Minute,
    Generator), with a "Unit State" column that is True when the generator
    is on. This pivots to a [G, T] matrix, selecting `hours` distinct
    (Date, Hour, Minute) timestamps starting at `start_hour`. By default
    (`hours=None`) it covers the entire year of the simulation.

    Returns
    -------
    generator_ids : np.ndarray [G]
        Generator names in row order.
    commitment : np.ndarray [G, 24, D]
        1 = on, 0 = off, reshaped into 24 hours x D days (e.g. D=366 for a
        full leap year) whenever the selected window is a whole number of
        days; otherwise a flat [G, T] array.
    """
    df = pd.read_csv(thermal_detail_csv)
    df["Unit State"] = df["Unit State"].astype(bool).astype(int)

    timestamps = df[["Date", "Hour", "Minute"]].drop_duplicates().sort_values(["Date", "Hour", "Minute"])
    end = None if hours is None else start_hour + hours
    window = timestamps.iloc[start_hour:end]
    df = df.merge(window, on=["Date", "Hour", "Minute"], how="inner")

    df["_t"] = df.groupby(["Date", "Hour", "Minute"]).ngroup()
    pivot = df.pivot(index="Generator", columns="_t", values="Unit State")

    generator_ids = pivot.index.to_numpy()
    commitment = pivot.to_numpy()

    n_hours = commitment.shape[1]
    if n_hours % 24 == 0:
        n_days = n_hours // 24
        commitment = commitment.reshape(len(generator_ids), n_days, 24).transpose(0, 2, 1)

    return generator_ids, commitment


def build_dataset(
    hours: int | None = 24,
    start_hour: int = 0,
    commitment_hours: int | None = None,
    include_renewable: bool = False,
):
    """Assemble topology, node features, edge features, and commitment status into one dict.

    `hours`/`start_hour` control the NF (nodal demand) window (default:
    first 24 hours, day-ahead; pass `hours=None` for the full year).
    `commitment_hours` controls the commitment status window independently;
    `None` (default) pulls the full year.

    When `include_renewable` is True, the hourly available renewable
    generation per bus is loaded over the same window and returned under the
    "NR" key, shaped exactly like "NF". It is kept as a separate array rather
    than folded into "NF" so that existing callers indexing "NF" as a pure
    demand profile keep working; downstream graph builders can stack the two
    into a 2-channel node feature. "has_renewable" records whether the key
    is present.
    """
    bus_ids, bus_index, edge_index, adjacency = load_topology()
    NF = load_node_features(bus_ids, hours=hours, start_hour=start_hour)
    EF = load_edge_features()
    generator_ids, commitment = load_commitment_status(hours=commitment_hours, start_hour=0)
    gen_bus_idx = generator_bus_map(generator_ids, bus_index)

    dataset = {
        "bus_ids": bus_ids,
        "bus_index": bus_index,
        "edge_index": edge_index,
        "adjacency": adjacency,
        "NF": NF,
        "EF": EF,
        "generator_ids": generator_ids,
        "commitment": commitment,
        "gen_bus_idx": gen_bus_idx,
        "has_renewable": include_renewable,
    }

    if include_renewable:
        dataset["NR"] = load_renewable_features(bus_ids, hours=hours, start_hour=start_hour)

    return dataset


if __name__ == "__main__":
    data = build_dataset(include_renewable=True)
    print(f"N (buses)  = {len(data['bus_ids'])}")
    print(f"E (lines)  = {data['edge_index'].shape[1]}")
    print(f"NF shape   = {data['NF'].shape}")
    print(f"NR shape   = {data['NR'].shape}")
    print(f"EF shape   = {data['EF'].shape}")
    print(f"G (gens)   = {len(data['generator_ids'])}")
    print(f"commitment shape = {data['commitment'].shape}")
