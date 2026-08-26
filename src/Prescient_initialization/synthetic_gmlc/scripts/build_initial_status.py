"""
Build RTS_Data/SourceData/initial_status.csv from gen.csv.

Format per the Prescient docs:
https://prescient.readthedocs.io/en/latest/reference/file_formats/rts-gmlc/initial_status.html

    - The header row has one column per generator, named by GEN UID from gen.csv.
      Every generator in gen.csv gets a column (renewables and storage included).
    - Row 1 (required): status -- the number of time periods the unit has been
      running (positive) or has been shut down (negative) at the start of the
      simulation. E.g. +168 = on for the last 168 periods, -24 = off for 24.
    - Row 2 (optional): power output in the period preceding the simulation.
      Must be populated for every generator or left blank entirely.
    - Row 3 (optional): reactive power in the preceding period. Only allowed if
      row 2 is populated. Not written here.

Default policy (deterministic "typical day" warm start, not yet randomized):
    - NUCLEAR and HYDRO/ROR default ON at ON_PERIODS, nuclear at PMax and hydro
      at a nominal 50 MW -- matching the reference RTS-GMLC initial_status.csv.
    - STEAM / CC default ON at ON_PERIODS and PMin MW (a conservative,
      always-feasible floor above their minimum stable level).
    - CT peakers default OFF at -OFF_PERIODS with 0 MW.
    - SYNC_COND default ON but carry no real power, so 0 MW.
    - Renewables (PV, RTPV, WIND, CSP) and STORAGE default OFF at -OFF_PERIODS
      with 0 MW; their actual output is driven by the timeseries, not by this
      initial condition.

Alongside the CSV, this writes a JSON template of the same data, in the
{"gen_name": [status, power_output]} format that update_gmlc.py's
update_initial_status() consumes. Edit that JSON (or generate your own with the
same schema) and feed it back through update_gmlc.py to set a different
initial-condition scenario.
"""

import csv
import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "RTS_Data", "SourceData")
GEN_CSV = os.path.join(SOURCE_DATA_DIR, "gen.csv")
OUT_CSV = os.path.join(SOURCE_DATA_DIR, "initial_status.csv")
OUT_JSON = os.path.join(SCRIPT_DIR, "..", "initial_status.json")

ON_PERIODS = 168     # periods a default-on unit has already been running
OFF_PERIODS = 24     # periods a default-off unit has already been down

# Unit types that start the horizon committed.
DEFAULT_ON_TYPES = {"STEAM", "CC", "NUCLEAR", "SYNC_COND", "HYDRO", "ROR"}

HYDRO_NOMINAL_MW = 50.0  # matches the reference RTS-GMLC initial_status.csv


def initial_condition(gen_row):
    """Return (status, power_output) for one row of gen.csv."""
    unit_type = gen_row["Unit Type"]

    if unit_type not in DEFAULT_ON_TYPES:
        return -OFF_PERIODS, 0.0

    if unit_type == "SYNC_COND":
        power = 0.0
    elif unit_type == "NUCLEAR":
        power = float(gen_row["PMax MW"])
    elif unit_type in ("HYDRO", "ROR"):
        power = HYDRO_NOMINAL_MW
    else:  # STEAM, CC
        power = float(gen_row["PMin MW"])

    return ON_PERIODS, power


def build_initial_status(gen_csv=GEN_CSV, out_csv=OUT_CSV, out_json=OUT_JSON):
    gen = pd.read_csv(gen_csv)

    gen_names, statuses, powers = [], [], []
    for _, g in gen.iterrows():
        status, power = initial_condition(g)
        gen_names.append(g["GEN UID"])
        statuses.append(status)
        powers.append(power)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(gen_names)
        writer.writerow(statuses)
        writer.writerow(powers)

    n_on = sum(1 for s in statuses if s > 0)
    print(f"Wrote {out_csv} ({len(gen_names)} generators, {n_on} initially on)")

    json_data = {name: [s, p] for name, s, p in zip(gen_names, statuses, powers)}
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Wrote {out_json} ({len(json_data)} generators)")

    return gen_names, statuses, powers


if __name__ == "__main__":
    build_initial_status()
