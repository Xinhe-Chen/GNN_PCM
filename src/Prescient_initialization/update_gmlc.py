"""
Update synthetic_gmlc timeseries_data_files from a JSON profile file.

The JSON file holds a dict of dicts:
    {
      "Load":  {"1": [...], "2": [...], "3": [...]},
      "PV":    {"320_PV_1": [...], ...},
      "WIND":  {"309_WIND_1": [...], ...},
      "RTPV":  {"308_RTPV_1": [...], ...},
      "Hydro": {"122_HYDRO_1": [...], ...},
    }

Each time series is hourly and must be exactly as long as the DAY_AHEAD file's
existing row count (48 = 1 PCM day + 1 UC look-ahead day, by default).

Only categories/columns present in the JSON are overwritten; everything else in
the DAY_AHEAD files is left untouched.

Since the PCM only ever consumes the first 5-minute period of each hour from the
REAL_TIME data (real-time clearing runs at 60-minute resolution here), REAL_TIME
files are not read from the JSON at all -- they are rebuilt directly from the
(possibly just-updated) DAY_AHEAD files by repeating each hourly value across all
5-minute periods (12 periods/hour) in that hour.

Separately, update_initial_status() updates RTS_Data/SourceData/initial_status.csv
(see synthetic_gmlc/scripts/build_initial_status.py) from a JSON file of the form:
    {"gen_name": [status, power_generated], ...}
where status is signed hours in the current state (+on/-off) and power_generated
is the MW output at t=0. Only generators present in the JSON are updated, and
only committable generators (Committable == "Yes" in the CSV) may be updated.
"""

import argparse
import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TS_ROOT = os.path.join(SCRIPT_DIR, "synthetic_gmlc", "RTS_Data", "timeseries_data_files")
DEFAULT_JSON_PATH = os.path.join(SCRIPT_DIR, "synthetic_gmlc", "synthetic_profiles.json")
DEFAULT_INITIAL_STATUS_CSV = os.path.join(SCRIPT_DIR, "synthetic_gmlc", "RTS_Data", "SourceData", "initial_status.csv")
DEFAULT_INITIAL_STATUS_JSON = os.path.join(SCRIPT_DIR, "synthetic_gmlc", "initial_status.json")

# category -> (subdir, base_name)
CATEGORY_MAP = {
    "Load": ("Load", "regional_Load"),
    "PV": ("PV", "pv"),
    "WIND": ("WIND", "wind"),
    "RTPV": ("RTPV", "rtpv"),
    "Hydro": ("Hydro", "hydro"),
}

META_COLS = ["Year", "Month", "Day", "Period"]
REAL_TIME_PERIODS_PER_DAY = 288
DAY_AHEAD_PERIODS_PER_DAY = 24
PERIODS_PER_HOUR_RT = REAL_TIME_PERIODS_PER_DAY // DAY_AHEAD_PERIODS_PER_DAY  # 12


def load_profiles(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def update_day_ahead(da_path, series_dict, category):
    df = pd.read_csv(da_path)
    n_rows = len(df)

    for gen_name, series in series_dict.items():
        if gen_name not in df.columns:
            raise KeyError(
                f"[{category}] column '{gen_name}' not found in {da_path}. "
                f"Available columns: {[c for c in df.columns if c not in META_COLS]}"
            )
        if len(series) != n_rows:
            raise ValueError(
                f"[{category}/{gen_name}] time series has {len(series)} points, "
                f"but {da_path} has {n_rows} hourly rows (Period 1-{DAY_AHEAD_PERIODS_PER_DAY} "
                f"x N days). Provide exactly {n_rows} hourly values."
            )
        df[gen_name] = series

    df.to_csv(da_path, index=False)
    return df


def rebuild_real_time(rt_path, day_ahead_df):
    """Expand each DAY_AHEAD hourly row into PERIODS_PER_HOUR_RT identical 5-min rows."""
    data_cols = [c for c in day_ahead_df.columns if c not in META_COLS]

    rows = []
    for _, row in day_ahead_df.iterrows():
        year, month, day = row["Year"], row["Month"], row["Day"]
        for sub_period in range(PERIODS_PER_HOUR_RT):
            rt_period = int((row["Period"] - 1) * PERIODS_PER_HOUR_RT + sub_period + 1)
            rows.append([year, month, day, rt_period] + [row[c] for c in data_cols])

    rt_df = pd.DataFrame(rows, columns=META_COLS + data_cols)
    rt_df[["Year", "Month", "Day", "Period"]] = rt_df[["Year", "Month", "Day", "Period"]].astype(int)
    rt_df.to_csv(rt_path, index=False)
    return rt_df


def update_gmlc(json_path, ts_root=DEFAULT_TS_ROOT):
    profiles = load_profiles(json_path)

    unknown = set(profiles) - set(CATEGORY_MAP)
    if unknown:
        raise KeyError(f"Unknown categories in JSON: {sorted(unknown)}. Expected one of {list(CATEGORY_MAP)}")

    for category, series_dict in profiles.items():
        subdir, base_name = CATEGORY_MAP[category]
        da_path = os.path.join(ts_root, subdir, f"DAY_AHEAD_{base_name}.csv")
        rt_path = os.path.join(ts_root, subdir, f"REAL_TIME_{base_name}.csv")

        day_ahead_df = update_day_ahead(da_path, series_dict, category)
        rebuild_real_time(rt_path, day_ahead_df)
        print(f"Updated {da_path} and {rt_path}")


def update_initial_status(json_path, csv_path=DEFAULT_INITIAL_STATUS_CSV):
    """Update initial_status.csv's status/power_generated columns from a JSON file
    of the form {"gen_name": [status, power_generated], ...}. Only generators
    present in the JSON are touched, and they must already be Committable == "Yes"
    in the CSV (status/power_generated aren't meaningful for non-committable units).
    """
    with open(json_path, "r") as f:
        updates = json.load(f)

    df = pd.read_csv(csv_path)
    df = df.set_index("GEN UID", drop=False)

    for gen_name, (status, power_generated) in updates.items():
        if gen_name not in df.index:
            raise KeyError(
                f"'{gen_name}' not found in {csv_path}. "
                f"Available generators: {list(df.index)}"
            )
        if df.loc[gen_name, "Committable"] != "Yes":
            raise ValueError(
                f"'{gen_name}' is not Committable in {csv_path}; "
                f"status/power_generated are not meaningful for it."
            )
        df.loc[gen_name, "status"] = status
        df.loc[gen_name, "power_generated"] = power_generated

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} ({len(updates)} generators)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Update synthetic_gmlc timeseries and/or initial status from JSON files.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help="Path to the JSON file with the new time series (default: synthetic_gmlc/synthetic_profiles.json).",
    )
    parser.add_argument(
        "--ts-root",
        default=DEFAULT_TS_ROOT,
        help="Path to timeseries_data_files root (default: synthetic_gmlc/RTS_Data/timeseries_data_files).",
    )
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Skip the timeseries update (useful if you only want --initial-status-json).",
    )
    parser.add_argument(
        "--initial-status-json",
        default=None,
        help="Path to a JSON file of {\"gen_name\": [status, power_generated]} to also/instead "
             "update initial_status.csv.",
    )
    parser.add_argument(
        "--initial-status-csv",
        default=DEFAULT_INITIAL_STATUS_CSV,
        help="Path to initial_status.csv (default: synthetic_gmlc/RTS_Data/SourceData/initial_status.csv).",
    )
    args = parser.parse_args()

    if not args.skip_timeseries:
        update_gmlc(args.json_path, args.ts_root)

    if args.initial_status_json:
        update_initial_status(args.initial_status_json, args.initial_status_csv)


if __name__ == "__main__":
    main()
