"""
example_prescient_init.py

Generate an `initial_status.csv` file for a Prescient / RTS-GMLC production
cost model run.

File format (see
https://prescient.readthedocs.io/en/latest/reference/file_formats/rts-gmlc/initial_status.html):

    header : one column per generator, named as in the GEN UID column of gen.csv
    row 1  : initial status. positive = periods the unit has been running,
             negative = periods since it was shut down. Required for all gens.
    row 2  : power output (MW) in the period just before the simulation starts.
             Blank for all generators, or populated for all.
    row 3  : reactive power (MVAR) just before the start. May only hold data
             if row 2 also holds data.

Drop the resulting file in the same source-data directory as gen.csv, bus.csv,
branch.csv, etc.

Usage:
    python example_prescient_init.py --gen-csv path/to/gen.csv \
                                     --out path/to/initial_status.csv
"""

import argparse

import pandas as pd

# Fuel types treated as "always on" baseload at the start of the run.
ALWAYS_ON_FUELS = ("Nuclear", "Hydro")

# How long units are assumed to have been in their current state, in time
# periods. Make the magnitude larger than any min up/down time in gen.csv if
# you do not want the first RUC constrained by unit history.
ON_PERIODS = 168
OFF_PERIODS = -24


def build_initial_status(gen: pd.DataFrame):
    """Return (names, status, power) lists, one entry per generator."""
    names, status, power = [], [], []

    for _, row in gen.iterrows():
        names.append(row["GEN UID"])
        if row["Fuel"] in ALWAYS_ON_FUELS:
            status.append(ON_PERIODS)
            power.append(float(row["PMax MW"]))
        else:
            status.append(OFF_PERIODS)
            power.append(0.0)

    return names, status, power


def validate(gen: pd.DataFrame, status, power):
    """Catch the initial conditions that most often make the first RUC infeasible."""
    problems = []

    for i, (_, row) in enumerate(gen.iterrows()):
        name, s, p = row["GEN UID"], status[i], power[i]
        pmin, pmax = float(row["PMin MW"]), float(row["PMax MW"])

        if s == 0:
            problems.append(f"{name}: status of 0 is ambiguous; use a nonzero value")
        elif s < 0 and p != 0.0:
            problems.append(f"{name}: unit is off (status {s}) but output is {p} MW")
        elif s > 0 and not (pmin <= p <= pmax):
            problems.append(
                f"{name}: unit is on (status {s}) but output {p} MW is outside "
                f"[{pmin}, {pmax}]"
            )

    return problems


def write_initial_status(path, names, status, power=None, reactive=None):
    """Write the file. Omits row 2 and 3 entirely when not supplied."""
    if reactive is not None and power is None:
        raise ValueError("row 3 (reactive power) requires row 2 (power output)")

    with open(path, "w") as f:
        f.write(",".join(str(n) for n in names) + "\n")
        f.write(",".join(str(s) for s in status) + "\n")
        if power is not None:
            f.write(",".join(str(p) for p in power) + "\n")
        if reactive is not None:
            f.write(",".join(str(q) for q in reactive) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-csv", default="../../data/GMLC_source_data/gen.csv", help="path to gen.csv")
    parser.add_argument("--out", default="../../data/GMLC_source_data/initial_status.csv", help="output path")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="write only the status row, letting Prescient infer initial output",
    )
    args = parser.parse_args()

    gen = pd.read_csv(args.gen_csv)
    names, status, power = build_initial_status(gen)

    for problem in validate(gen, status, power):
        print(f"WARNING  {problem}")

    write_initial_status(
        args.out, names, status, power=None if args.status_only else power
    )
    print(f"Wrote {args.out} with {len(names)} generators.")


if __name__ == "__main__":
    main()