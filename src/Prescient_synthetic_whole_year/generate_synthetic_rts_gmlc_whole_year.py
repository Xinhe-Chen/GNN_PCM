"""Generate a synthetic whole-year RTS-GMLC case for Prescient.

Copies ``origin_rts_gmlc`` to ``{suffix}_rts_gmlc`` and perturbs the day-ahead load
and renewable time series with a multiplicative AR(1) coefficient:

    x_t = xhat_t + f_t * xhat_t = (1 + f_t) * xhat_t

    f_t = alpha * f_{t-1} + alpha * eps_t,   f_0 = 0,   eps_t ~ N(0, eps_std)

Each series (each region, each generator) draws its own independent ``f`` path.
Because |f| << 1 the perturbed series is never negative, and because the
perturbation is multiplicative an hour that is exactly zero in the original stays
exactly zero -- so PV/RTPV keep their true overnight zeros.

The real-time files are then rebuilt from the synthetic day-ahead values: the
simulation is run on a 60-minute real-time step, so every sub-hourly period within
an hour is set to that hour's synthetic day-ahead value. The row layout of the
original real-time files (5-minute periods) is preserved.

Everything needed to reproduce the case -- alpha, eps_std, seed, and the per-column
draw order -- is written to ``synthetic_config.json`` inside the new case folder.

Usage
-----
    python generate_synthetic_rts_gmlc_whole_year.py --suffix case01
    python generate_synthetic_rts_gmlc_whole_year.py --suffix hi_var --alpha 0.9 --eps-std 0.25
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# case layout
# --------------------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ORIGIN_DIRNAME = "origin_rts_gmlc"

INDEX_COLS = ["Year", "Month", "Day", "Period"]

# name -> (day-ahead file, real-time file), both relative to timeseries_data_files/
SERIES_FILES = {
    "Load": ("Load/DAY_AHEAD_regional_Load.csv", "Load/REAL_TIME_regional_Load.csv"),
    "Hydro": ("Hydro/DAY_AHEAD_hydro.csv", "Hydro/REAL_TIME_hydro.csv"),
    "PV": ("PV/DAY_AHEAD_pv.csv", "PV/REAL_TIME_pv.csv"),
    "RTPV": ("RTPV/DAY_AHEAD_rtpv.csv", "RTPV/REAL_TIME_rtpv.csv"),
    "WIND": ("WIND/DAY_AHEAD_wind.csv", "WIND/REAL_TIME_wind.csv"),
}


# --------------------------------------------------------------------------------------
# the perturbation model
# --------------------------------------------------------------------------------------

def simulate_f(n: int, alpha: float, eps_std: float, rng: np.random.Generator) -> np.ndarray:
    """f_t = alpha * f_{t-1} + alpha * eps_t, with f_0 = 0 and eps_t ~ N(0, eps_std)."""
    eps = rng.normal(0.0, eps_std, size=n)
    f = np.empty(n, dtype=float)
    prev = 0.0
    for t in range(n):
        prev = alpha * prev + alpha * eps[t]
        f[t] = prev
    return f


def perturb_frame(df: pd.DataFrame, value_cols: list[str], alpha: float,
                  eps_std: float, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Apply x = (1 + f) * xhat to every value column, each with its own f path."""
    out = df.copy()
    n = len(df)
    stats = {}

    for col in value_cols:
        f = simulate_f(n, alpha, eps_std, rng)
        original = df[col].to_numpy(dtype=float)
        perturbed = original + f * original
        out[col] = perturbed

        stats[col] = {
            "f_min": float(f.min()),
            "f_max": float(f.max()),
            "f_std": float(f.std()),
            "original_sum": float(original.sum()),
            "synthetic_sum": float(perturbed.sum()),
            "synthetic_min": float(perturbed.min()),
            "synthetic_max": float(perturbed.max()),
        }

    return out, stats


def expand_day_ahead_to_real_time(da: pd.DataFrame, rt_template: pd.DataFrame,
                                  value_cols: list[str]) -> pd.DataFrame:
    """Fill a real-time file from hourly day-ahead values.

    The real-time files carry sub-hourly periods (288/day = 5 minutes). The
    simulation uses a 60-minute real-time step, so every period inside an hour takes
    that hour's synthetic day-ahead value. The template's row layout is preserved
    exactly; only the value columns are replaced.
    """
    periods_per_day = int(rt_template.groupby(["Year", "Month", "Day"])["Period"].size().max())
    if periods_per_day % 24:
        raise ValueError(
            f"real-time file has {periods_per_day} periods/day, not a multiple of 24")
    periods_per_hour = periods_per_day // 24

    out = rt_template.copy()
    # day-ahead Period is the 1-based hour of day; real-time Period is the 1-based
    # sub-hourly slot, so map it down to the hour it falls in.
    rt_hour = (out["Period"].to_numpy() - 1) // periods_per_hour + 1

    da_keyed = da.set_index(INDEX_COLS)
    key = pd.MultiIndex.from_arrays(
        [out["Year"].to_numpy(), out["Month"].to_numpy(), out["Day"].to_numpy(), rt_hour],
        names=INDEX_COLS,
    )
    missing = ~key.isin(da_keyed.index)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} real-time rows have no matching day-ahead hour")

    for col in value_cols:
        out[col] = da_keyed[col].reindex(key).to_numpy()

    return out, periods_per_hour


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def warn_if_perturbation_too_large(alpha: float, eps_std: float) -> None:
    """Flag hyperparameters that can drive (1 + f) below zero.

    f is AR(1) with lag coefficient alpha and innovation sd alpha * eps_std, so its
    stationary sd is alpha * eps_std / sqrt(1 - alpha**2). Once that approaches 1,
    f can dip under -1 and the perturbed series goes negative -- which is not a
    meaningful load or generation profile. The run still aborts on the actual check
    downstream; this just says so before spending time on the copy.
    """
    if alpha >= 1.0:
        print(f"WARNING: alpha={alpha} makes f non-stationary (it will drift without bound)")
        return
    f_std = alpha * eps_std / np.sqrt(1.0 - alpha ** 2)
    print(f"       implied stationary std(f) = {f_std:.4f} "
          f"(perturbation is roughly +/-{100 * f_std:.2f}% of the original)")
    if f_std > 0.25:
        print(f"WARNING: std(f)={f_std:.3f} is large; f may fall below -1 and drive the "
              f"series negative, which aborts the run. Reduce alpha or eps_std.")


def build_case(suffix: str, alpha: float, eps_std: float, seed: int,
               origin: Path, output_root: Path, force: bool) -> Path:
    if not origin.is_dir():
        raise FileNotFoundError(f"origin case not found: {origin}")

    dest = output_root / f"{suffix}_rts_gmlc"
    if dest.exists():
        if not force:
            raise FileExistsError(
                f"{dest} already exists; pass --force to overwrite it")
        print(f"[1/4] removing existing {dest.name}")
        shutil.rmtree(dest)

    warn_if_perturbation_too_large(alpha, eps_std)

    print(f"[1/4] copying {origin.name} -> {dest.name}")
    shutil.copytree(origin, dest)
    try:
        return _populate_case(dest, origin, suffix, alpha, eps_std, seed)
    except BaseException:
        # never leave a half-perturbed case behind -- it would look usable
        print(f"\nfailed; removing incomplete {dest.name}")
        shutil.rmtree(dest, ignore_errors=True)
        raise


def _populate_case(dest: Path, origin: Path, suffix: str, alpha: float,
                   eps_std: float, seed: int) -> Path:

    ts_dir = dest / "timeseries_data_files"
    rng = np.random.default_rng(seed)
    record: dict[str, dict] = {}

    print(f"[2/4] perturbing day-ahead series (alpha={alpha}, eps_std={eps_std}, seed={seed})")
    day_ahead: dict[str, pd.DataFrame] = {}
    for name, (da_rel, _) in SERIES_FILES.items():
        da_path = ts_dir / da_rel
        da = pd.read_csv(da_path)
        value_cols = [c for c in da.columns if c not in INDEX_COLS]

        synth, stats = perturb_frame(da, value_cols, alpha, eps_std, rng)

        # invariants the model guarantees
        vals = synth[value_cols].to_numpy(dtype=float)
        orig_vals = da[value_cols].to_numpy(dtype=float)
        if (vals < 0).any():
            raise ValueError(f"{name}: perturbation produced negative values")
        if not np.array_equal(vals[orig_vals == 0], orig_vals[orig_vals == 0]):
            raise ValueError(f"{name}: zero-valued hours were perturbed")

        synth.to_csv(da_path, index=False)
        day_ahead[name] = synth
        record[name] = {
            "day_ahead_file": da_rel,
            "n_columns": len(value_cols),
            "n_hours": len(synth),
            "columns": stats,
        }
        print(f"       {name:6s} {len(value_cols):3d} series x {len(synth)} hours")

    print("[3/4] rebuilding real-time series from synthetic day-ahead values")
    for name, (_, rt_rel) in SERIES_FILES.items():
        rt_path = ts_dir / rt_rel
        rt = pd.read_csv(rt_path)
        value_cols = [c for c in rt.columns if c not in INDEX_COLS]

        da = day_ahead[name]
        da_cols = [c for c in da.columns if c not in INDEX_COLS]
        if value_cols != da_cols:
            raise ValueError(
                f"{name}: real-time columns do not match day-ahead columns")

        rt_out, per_hour = expand_day_ahead_to_real_time(da, rt, value_cols)
        rt_out.to_csv(rt_path, index=False)

        record[name]["real_time_file"] = rt_rel
        record[name]["real_time_periods_per_hour"] = per_hour
        record[name]["n_real_time_rows"] = len(rt_out)
        print(f"       {name:6s} {len(rt_out)} rows ({per_hour} periods/hour -> held flat)")

    print("[4/4] writing synthetic_config.json")
    config = {
        "suffix": suffix,
        "case_name": dest.name,
        "origin_case": str(origin),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "day_ahead": "x_t = xhat_t + f_t * xhat_t = (1 + f_t) * xhat_t",
            "coefficient": "f_t = alpha * f_{t-1} + alpha * eps_t, f_0 = 0, eps_t ~ N(0, eps_std)",
            "real_time": (
                "each real-time period within an hour is held at that hour's "
                "synthetic day-ahead value (60-minute real-time step)"
            ),
        },
        "hyperparameters": {
            "alpha": alpha,
            "eps_std": eps_std,
            "eps_variance": eps_std ** 2,
            "seed": seed,
        },
        "draw_order": {
            "note": (
                "f paths are drawn from a single numpy default_rng(seed), one per column, "
                "iterating sources in this order and columns in file order"
            ),
            "sources": list(SERIES_FILES),
        },
        "series": record,
    }
    config_path = dest / "synthetic_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\ndone: {dest}")
    print(f"      config: {config_path}")
    return dest


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a synthetic whole-year RTS-GMLC Prescient case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suffix", required=True,
                   help="case is written to '{suffix}_rts_gmlc'")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="AR(1) coefficient, scaling both the lag term and the innovation")
    p.add_argument("--eps-std", type=float, default=0.1,
                   help="standard deviation of the innovation eps ~ N(0, eps_std)")
    p.add_argument("--eps-var", type=float, default=None,
                   help="variance of eps; alternative to --eps-std (mutually exclusive)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the random number generator, for reproducibility")
    p.add_argument("--origin", type=Path, default=SCRIPT_DIR / ORIGIN_DIRNAME,
                   help="the pristine case to copy from")
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR,
                   help="directory the new case folder is created in")
    p.add_argument("--force", action="store_true",
                   help="overwrite the destination case folder if it already exists")

    args = p.parse_args(argv)

    if args.eps_var is not None:
        if "--eps-std" in (argv if argv is not None else __import__("sys").argv):
            p.error("pass either --eps-std or --eps-var, not both")
        if args.eps_var < 0:
            p.error("--eps-var must be non-negative")
        args.eps_std = float(np.sqrt(args.eps_var))
    if args.eps_std < 0:
        p.error("--eps-std must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    build_case(
        suffix=args.suffix,
        alpha=args.alpha,
        eps_std=args.eps_std,
        seed=args.seed,
        origin=args.origin.resolve(),
        output_root=args.output_root.resolve(),
        force=args.force,
    )


if __name__ == "__main__":
    main()
