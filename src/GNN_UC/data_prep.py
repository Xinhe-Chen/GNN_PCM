"""
Data preparation for GNN-based unit commitment / SCUC modeling.

Builds, from the GMLC source data:
    1. Base topology: bus list (nodes), line list (edges), adjacency matrix / edge index.
    2. Node features NF [N, T]: 24-hour nodal demand profile for every bus.
    3. Edge features EF [E, 2]: line susceptance and thermal (continuous) rating.
    4. Generator commitment status [G, T]: on/off (1/0) for every generator, every hour.
    5. Optional node features NR [N, T]: hourly available renewable generation per
       bus (enabled with `build_dataset(include_renewable=True)`).
    6. Per-generator static features GF [G, K]: nameplate attributes from gen.csv
       (on by default, `build_dataset(include_static=True)`). Unlike NF/NR these
       are indexed by generator rather than by bus and do not vary over time.

Commitment labels can be merged across several PCM simulation runs (see
`build_dataset(pcm_runs=...)`), which appends each run's days as additional
training samples.

Source files (relative to repo root):
    data/GMLC_source_data/bus.csv
    data/GMLC_source_data/branch.csv
    data/GMLC_source_data/bus_load.csv
    data/GMLC_source_data/bus_renewable.csv
    data/GMLC_source_data/gen.csv
    data/PCM_results/<run>/thermal_detail.csv
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "GMLC_source_data"
BUS_CSV = DATA_DIR / "bus.csv"
BRANCH_CSV = DATA_DIR / "branch.csv"
BUS_LOAD_CSV = DATA_DIR / "bus_load.csv"
BUS_RENEWABLE_CSV = DATA_DIR / "bus_renewable.csv"
GEN_CSV = DATA_DIR / "gen.csv"

PCM_RESULTS_DIR = REPO_ROOT / "data" / "PCM_results"
DEFAULT_PCM_RUN = "base_case_pcm_test"
THERMAL_DETAIL_CSV = PCM_RESULTS_DIR / DEFAULT_PCM_RUN / "thermal_detail.csv"


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


#: Unit types one-hot encoded by `load_generator_features`, in a fixed order so
#: the column layout is stable regardless of which units appear in a given case.
UNIT_TYPES = ("CC", "CT", "NUCLEAR", "STEAM", "SYNC_COND")


def load_generator_features(generator_ids: np.ndarray, gen_csv: Path = GEN_CSV):
    """Extract per-generator static (nameplate) attributes.

    These are the unit characteristics that determine commitment behaviour but
    are absent from the demand/renewable node features: without them every
    generator sharing a bus receives an identical feature vector and is
    automorphic in the generator graph, so no GNN can tell a peaking CT from a
    nuclear unit at the same bus.

    Columns produced (in order):
      unit_type=<T> : one-hot over UNIT_TYPES
      log_pmax_mw, log_pmin_mw : log1p-scaled capacity (spans 0-400 MW)
      pmin_pmax_ratio          : must-run fraction; 0 for zero-capacity units
      min_up_time_hr, min_down_time_hr : commitment inertia
      ramp_rate_mw_per_min
      log_start_cost           : log1p-scaled hot-start cost (start fuel burn
                                 priced out, plus any non-fuel start cost)
      marginal_cost            : Fuel Price * HR_avg_0 / 1000 + VOM, the
                                 merit-order position that actually drives
                                 which units get committed

    Only nameplate attributes are used. Nothing here is derived from the PCM
    results -- a feature such as a unit's observed capacity factor would leak
    the label (a per-generator majority lookup alone scores ~0.91).

    Parameters
    ----------
    generator_ids : np.ndarray [G]
        Generator order to align to, as returned by `load_commitment_status`.
        gen.csv is keyed by "GEN UID" in its own order, so it is reindexed onto
        this list; using file order instead would silently pair each generator
        with another unit's attributes.

    Returns
    -------
    GF : np.ndarray [G, K], float32
    feature_names : list[str], length K
    """
    gen_df = pd.read_csv(gen_csv).set_index("GEN UID")

    missing = [g for g in generator_ids if g not in gen_df.index]
    if missing:
        raise KeyError(f"{len(missing)} generator(s) absent from {gen_csv.name}: {missing[:5]}")
    gen_df = gen_df.loc[generator_ids]

    def col(name):
        return pd.to_numeric(gen_df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    pmax, pmin = col("PMax MW"), col("PMin MW")
    unknown_types = set(gen_df["Unit Type"]) - set(UNIT_TYPES)
    if unknown_types:
        raise ValueError(f"Unit types not covered by UNIT_TYPES: {sorted(unknown_types)}")

    columns, feature_names = [], []

    for unit_type in UNIT_TYPES:
        columns.append((gen_df["Unit Type"] == unit_type).to_numpy(dtype=np.float64))
        feature_names.append(f"unit_type={unit_type}")

    # log1p compresses the heavy right tail of capacities and start costs so a
    # few large units do not dominate the z-scored feature.
    for values, name in [
        (np.log1p(pmax), "log_pmax_mw"),
        (np.log1p(pmin), "log_pmin_mw"),
        (np.divide(pmin, pmax, out=np.zeros_like(pmin), where=pmax > 0), "pmin_pmax_ratio"),
        (col("Min Up Time Hr"), "min_up_time_hr"),
        (col("Min Down Time Hr"), "min_down_time_hr"),
        (col("Ramp Rate MW/Min"), "ramp_rate_mw_per_min"),
        # "Non Fuel Start Cost $" is 0 for every thermal unit in this case, so
        # the fuel burned on a hot start carries the whole startup signal
        # (CT ~1.2k$, STEAM ~8.5k$, CC ~12.4k$) -- that asymmetry is what keeps
        # expensive-to-start units committed instead of cycling.
        (
            np.log1p(col("Start Heat Hot MBTU") * col("Fuel Price $/MMBTU") + col("Non Fuel Start Cost $")),
            "log_start_cost",
        ),
        (col("Fuel Price $/MMBTU") * col("HR_avg_0") / 1000.0 + col("VOM"), "marginal_cost"),
    ]:
        columns.append(values)
        feature_names.append(name)

    GF = np.stack(columns, axis=1).astype(np.float32)
    return GF, feature_names


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


def discover_pcm_runs(pcm_results_dir: Path = PCM_RESULTS_DIR):
    """List every PCM result folder that contains a thermal_detail.csv.

    Returns
    -------
    list[Path], sorted by folder name for reproducible ordering.
    """
    pcm_results_dir = Path(pcm_results_dir)
    runs = [p for p in sorted(pcm_results_dir.iterdir()) if p.is_dir() and (p / "thermal_detail.csv").is_file()]
    return runs


def resolve_pcm_runs(pcm_runs=None, pcm_results_dir: Path = PCM_RESULTS_DIR):
    """Normalize the `pcm_runs` argument into a list of run directories.

    Accepts:
      None    -> just the default base-case run (backward compatible)
      "all"   -> every run found by `discover_pcm_runs`
      str     -> a single run, by folder name or path
      sequence of str/Path -> those runs, in the given order
    """
    pcm_results_dir = Path(pcm_results_dir)

    if pcm_runs is None:
        candidates = [pcm_results_dir / DEFAULT_PCM_RUN]
    elif isinstance(pcm_runs, str) and pcm_runs == "all":
        candidates = discover_pcm_runs(pcm_results_dir)
        if not candidates:
            raise FileNotFoundError(f"No PCM runs with a thermal_detail.csv found under {pcm_results_dir}")
    elif isinstance(pcm_runs, (str, Path)):
        candidates = [Path(pcm_runs)]
    elif isinstance(pcm_runs, Sequence):
        candidates = [Path(r) for r in pcm_runs]
    else:
        raise TypeError(f"Unsupported `pcm_runs` value: {pcm_runs!r}")

    resolved = []
    for c in candidates:
        run_dir = c if c.is_absolute() or c.exists() else pcm_results_dir / c
        if not (run_dir / "thermal_detail.csv").is_file():
            available = [p.name for p in discover_pcm_runs(pcm_results_dir)]
            raise FileNotFoundError(
                f"No thermal_detail.csv in PCM run '{run_dir}'. Available runs: {available}"
            )
        resolved.append(run_dir)
    return resolved


def load_merged_commitment(run_dirs, hours: int | None = None, start_hour: int = 0):
    """Load and concatenate commitment labels from several PCM runs.

    Each run contributes its own days as additional samples along the last
    axis. All runs must expose the same generator list, in the same order,
    so that row g means the same unit everywhere.

    Returns
    -------
    generator_ids : np.ndarray [G]
    commitment : np.ndarray [G, 24, D_total]
        Days concatenated across runs, in `run_dirs` order.
    sample_day : np.ndarray [D_total]
        For each sample, its day index *within its own run*. This is the
        column to use when indexing the shared NF/NR arrays, which are
        indexed by calendar day rather than by sample.
    sample_run : np.ndarray [D_total]
        The run-folder name each sample came from.
    """
    generator_ids = None
    chunks, day_idx, run_names = [], [], []

    for run_dir in run_dirs:
        ids, commitment = load_commitment_status(
            Path(run_dir) / "thermal_detail.csv", hours=hours, start_hour=start_hour
        )
        if commitment.ndim != 3:
            raise ValueError(
                f"Run '{Path(run_dir).name}' did not resolve to whole days (got shape {commitment.shape}); "
                "merging requires a [G, 24, D] layout."
            )

        if generator_ids is None:
            generator_ids = ids
        elif not np.array_equal(ids, generator_ids):
            raise ValueError(
                f"Run '{Path(run_dir).name}' has a different generator list than '{Path(run_dirs[0]).name}'; "
                "cannot merge runs with mismatched units."
            )

        chunks.append(commitment)
        day_idx.append(np.arange(commitment.shape[2]))
        run_names.append(np.full(commitment.shape[2], Path(run_dir).name))

    return (
        generator_ids,
        np.concatenate(chunks, axis=2),
        np.concatenate(day_idx),
        np.concatenate(run_names),
    )


def build_dataset(
    hours: int | None = 24,
    start_hour: int = 0,
    commitment_hours: int | None = None,
    include_renewable: bool = False,
    pcm_runs=None,
    include_static: bool = True,
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

    `pcm_runs` selects which PCM simulation results supply the commitment
    labels: None (default) uses only the base case, "all" merges every run
    under data/PCM_results, and a name/path or list of them selects runs
    explicitly. Merging appends each run's days as extra samples, so
    "commitment" becomes [G, 24, D_total].

    IMPORTANT when merging: NF/NR come from the shared GMLC source data and
    are indexed by *calendar day*, while "commitment" is indexed by *sample*.
    Use "sample_day" to map a sample back to its NF/NR column. Because
    different runs of the same system share the same input profiles, the
    same day appears once per run with identical features but possibly
    different labels -- split by "sample_day" (not by sample) to keep the
    same day out of both train and test.

    When `include_static` is True (the default), per-generator nameplate
    attributes from gen.csv are returned under "GF" [G, K], with their column
    names under "gen_feature_names". Unlike NF/NR these are indexed by
    generator and constant over time, so a graph builder appends the same
    block to every sample. They are what makes generators at a shared bus
    distinguishable at all -- see `load_generator_features`.
    """
    bus_ids, bus_index, edge_index, adjacency = load_topology()
    NF = load_node_features(bus_ids, hours=hours, start_hour=start_hour)
    EF = load_edge_features()

    run_dirs = resolve_pcm_runs(pcm_runs)
    generator_ids, commitment, sample_day, sample_run = load_merged_commitment(
        run_dirs, hours=commitment_hours, start_hour=0
    )
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
        "has_static": include_static,
        "sample_day": sample_day,
        "sample_run": sample_run,
        "pcm_runs": [p.name for p in run_dirs],
    }

    if include_renewable:
        dataset["NR"] = load_renewable_features(bus_ids, hours=hours, start_hour=start_hour)

    if include_static:
        GF, gen_feature_names = load_generator_features(generator_ids)
        dataset["GF"] = GF
        dataset["gen_feature_names"] = gen_feature_names

    return dataset


if __name__ == "__main__":
    print(f"available PCM runs: {[p.name for p in discover_pcm_runs()]}\n")

    data = build_dataset(include_renewable=True, pcm_runs="all")
    print(f"N (buses)  = {len(data['bus_ids'])}")
    print(f"E (lines)  = {data['edge_index'].shape[1]}")
    print(f"NF shape   = {data['NF'].shape}")
    print(f"NR shape   = {data['NR'].shape}")
    print(f"EF shape   = {data['EF'].shape}")
    print(f"G (gens)   = {len(data['generator_ids'])}")
    print(f"commitment shape = {data['commitment'].shape}")
    print(f"GF shape   = {data['GF'].shape}")
    print(f"GF columns = {data['gen_feature_names']}")
    print(f"merged runs      = {data['pcm_runs']}")
    print(f"samples per run  = {dict(zip(*np.unique(data['sample_run'], return_counts=True)))}")
    print(f"distinct days    = {len(np.unique(data['sample_day']))}")
