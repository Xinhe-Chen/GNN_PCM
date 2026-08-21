"""
Build RTS_Data/SourceData/initial_status.csv from gen.csv.

This file is NOT part of stock RTS-GMLC / not documented in SourceData/README.md
(the README is stale even for the files it does cover -- e.g. storage.csv). It
follows the Prescient/Egret convention for generator initial conditions at the
start of a UC horizon:

    GEN UID         : matches gen.csv "GEN UID"
    Committable      : whether this generator carries a UC on/off decision
                        (thermal-type units with binding min up/down time).
                        Non-committable units (solar/wind/hydro/storage/sync
                        cond) are still listed for completeness but status /
                        power_generated are not meaningful for them.
    status           : hours the unit has been continuously in its current
                        state, SIGNED: positive = on for that many hours,
                        negative = off for that many hours. E.g. +12 means
                        "on for the last 12 hours"; -6 means "off for the
                        last 6 hours."
    power_generated  : MW output at t=0 (0 if off).

Default policy (deterministic "typical day" warm start, not yet randomized):
    - Baseload-ish committable units (STEAM, CC, NUCLEAR) default ON, with
      status = 10x their Min Up Time Hr (safely past any min-up requirement)
      and power_generated = PMin MW (a conservative, always-feasible floor).
    - Fast/peaking committable units (CT) default OFF, with
      status = -10x their Min Down Time Hr and power_generated = 0.
    - SYNC_COND units are committable (they do have on/off status) but carry
      no real power, so they default ON with power_generated = 0.
    - Non-committable units (PV, RTPV, WIND, HYDRO, ROR, CSP, STORAGE) get
      Committable=No, status/power_generated left blank -- their output is
      driven by the timeseries / dispatch, not a UC initial condition.

This is meant as a baseline/template: for scenario generation you'll likely
want to randomize status/power_generated (see the brainstormed sampling
ideas for generator initial conditions) rather than use this fixed baseline
directly.

Alongside the CSV, this also writes a JSON template of the same data, in the
{"gen_name": [status, power_generated]} format that update_gmlc.py's
update_initial_status() consumes -- only for committable generators, since
status/power_generated aren't meaningful for the rest. Edit this JSON (or
generate your own with the same schema) and feed it back through
update_gmlc.py to set a different initial-condition scenario.
"""

import json
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "RTS_Data", "SourceData")
GEN_CSV = os.path.join(SOURCE_DATA_DIR, "gen.csv")
OUT_CSV = os.path.join(SOURCE_DATA_DIR, "initial_status.csv")
OUT_JSON = os.path.join(SCRIPT_DIR, "..", "initial_status.json")

# Unit types that carry a real UC on/off commitment decision.
COMMITTABLE_TYPES = {"STEAM", "CC", "NUCLEAR", "CT", "SYNC_COND"}

# Within committable types, which default ON vs OFF for the baseline day.
DEFAULT_ON_TYPES = {"STEAM", "CC", "NUCLEAR", "SYNC_COND"}
DEFAULT_OFF_TYPES = {"CT"}

STATUS_HOURS_MULTIPLIER = 10  # multiple of min up/down time used for default status magnitude


def build_initial_status(gen_csv=GEN_CSV, out_csv=OUT_CSV, out_json=OUT_JSON):
    gen = pd.read_csv(gen_csv)

    rows = []
    for _, g in gen.iterrows():
        uid = g["GEN UID"]
        unit_type = g["Unit Type"]
        committable = unit_type in COMMITTABLE_TYPES

        if not committable:
            rows.append({"GEN UID": uid, "Committable": "No", "status": "", "power_generated": ""})
            continue

        if unit_type in DEFAULT_ON_TYPES:
            min_up = max(g["Min Up Time Hr"], 1.0)
            status = STATUS_HOURS_MULTIPLIER * min_up
            power_generated = 0.0 if unit_type == "SYNC_COND" else g["PMin MW"]
        else:  # DEFAULT_OFF_TYPES
            min_down = max(g["Min Down Time Hr"], 1.0)
            status = -STATUS_HOURS_MULTIPLIER * min_down
            power_generated = 0.0

        rows.append({
            "GEN UID": uid,
            "Committable": "Yes",
            "status": status,
            "power_generated": power_generated,
        })

    out = pd.DataFrame(rows, columns=["GEN UID", "Committable", "status", "power_generated"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} generators, {(out['Committable'] == 'Yes').sum()} committable)")

    committable = out[out["Committable"] == "Yes"]
    json_data = {
        row["GEN UID"]: [row["status"], row["power_generated"]]
        for _, row in committable.iterrows()
    }
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Wrote {out_json} ({len(json_data)} committable generators)")

    return out


if __name__ == "__main__":
    build_initial_status()
