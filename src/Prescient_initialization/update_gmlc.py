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
    {"gen_name": [status, power_output], ...}

That CSV follows the Prescient RTS-GMLC layout
(https://prescient.readthedocs.io/en/latest/reference/file_formats/rts-gmlc/initial_status.html):
one column per generator, named by GEN UID from gen.csv, with
    row 1 (required) : status -- periods the unit has been running (+) or off (-)
    row 2 (optional) : power output in the period preceding the simulation
    row 3 (optional) : reactive power in the preceding period
Only generators named in the JSON are updated; the rest keep their values.
"""

import argparse
import csv
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


def read_initial_status(csv_path=DEFAULT_INITIAL_STATUS_CSV):
    """Read initial_status.csv into (gen_names, status_row, power_row, reactive_row).

    Format (per Prescient docs): one column per generator, column name = GEN UID.
      row 1 (required) : status -- time periods running (+) or shut down (-)
      row 2 (optional) : power output in the period preceding the simulation
      row 3 (optional) : reactive power in the preceding period
    Optional rows come back as None when absent/blank.
    """
    with open(csv_path, "r", newline="") as f:
        rows = [r for r in csv.reader(f) if r]

    if not rows:
        raise ValueError(f"{csv_path} is empty")

    gen_names = rows[0]

    def row_or_none(idx):
        if len(rows) <= idx:
            return None
        row = rows[idx]
        if all(v.strip() == "" for v in row):
            return None
        if len(row) != len(gen_names):
            raise ValueError(
                f"{csv_path} row {idx + 1} has {len(row)} values but the header "
                f"names {len(gen_names)} generators."
            )
        return row

    status_row = row_or_none(1)
    if status_row is None:
        raise ValueError(f"{csv_path} is missing the required status row (row 2).")

    return gen_names, status_row, row_or_none(2), row_or_none(3)


def update_initial_status(json_path, csv_path=DEFAULT_INITIAL_STATUS_CSV):
    """Update initial_status.csv from a JSON file of the form
    {"gen_name": [status, power_output], ...}.

    Only generators named in the JSON are changed; every other column keeps its
    existing values. `status` is the number of time periods the unit has been
    running (positive) or shut down (negative); `power_output` is its output in
    the period immediately preceding the simulation.

    Per the Prescient format, the power row must be populated for every generator
    or left blank entirely -- so if the file currently has no power row, one is
    created with 0.0 for the generators the JSON doesn't mention.
    """
    with open(json_path, "r") as f:
        updates = json.load(f)

    gen_names, status_row, power_row, reactive_row = read_initial_status(csv_path)
    index = {name: i for i, name in enumerate(gen_names)}

    if power_row is None and updates:
        power_row = ["0.0"] * len(gen_names)

    for gen_name, values in updates.items():
        if gen_name not in index:
            raise KeyError(
                f"'{gen_name}' is not a column in {csv_path}. "
                f"Expected one of the {len(gen_names)} GEN UIDs from gen.csv."
            )
        if len(values) != 2:
            raise ValueError(
                f"'{gen_name}' maps to {values}; expected exactly "
                f"[status, power_output]."
            )
        status, power_output = values
        i = index[gen_name]
        status_row[i] = str(status)
        power_row[i] = str(power_output)

    out_rows = [gen_names, status_row]
    if power_row is not None:
        out_rows.append(power_row)
    if reactive_row is not None:
        out_rows.append(reactive_row)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)

    print(f"Updated {csv_path} ({len(updates)} of {len(gen_names)} generators)")
    return out_rows


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
