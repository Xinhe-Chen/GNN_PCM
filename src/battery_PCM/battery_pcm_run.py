"""Run a Prescient PCM on RTS-GMLC with a rule-based battery in the real-time market.

The day-ahead market is solved without the battery. In every real-time dispatch
(SCED) the battery charges when the DA LMP at its bus is below --buy-price and
discharges when it is above --sell-price, within its power and SoC limits. See
battery_plugin.py for the details.

Usage
-----
    # defaults: synthetic "test" case, 366 days, 50 MW / 200 MWh battery at bus 101
    python battery_pcm_run.py

    # a 1-day test at bus Abel (ID 101) with a 100 MW / 400 MWh battery
    python battery_pcm_run.py --num-days 1 --bus 101 \
        --p-charge-max 100 --p-discharge-max 100 --soc-max 400

    # another case / time window
    python battery_pcm_run.py --data-path ../Prescient_initialization/synthetic_gmlc/RTS_Data/SourceData \
        --start-date 01-01-2020 --num-days 1 --job-name init_case_battery

Results go to results/{job_name}_results/ next to this script: the usual
Prescient outputs plus battery_results.csv (one row per SCED step).
"""

import argparse
import os

from prescient.simulator import Prescient

this_file_path = os.path.dirname(os.path.realpath(__file__))

DEFAULT_DATA_PATH = os.path.join(
    this_file_path, "..", "Prescient_synthetic_whole_year", "origin_rts_gmlc", "SourceData")
PLUGIN_PATH = os.path.join(this_file_path, "battery_plugin.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prescient PCM with a rule-based real-time battery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- case / simulation window ---
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="RTS-GMLC SourceData folder")
    p.add_argument("--start-date", default="01-01-2020", help="MM-DD-YYYY")
    p.add_argument("--num-days", type=int, default=366)
    p.add_argument("--job-name", default=None,
                   help="labels the output folder (default: battery_<bus>)")
    p.add_argument("--solver", default="gurobi",
                   help="solver for both the RUC and the SCED")

    # --- battery ---
    b = p.add_argument_group("battery")
    b.add_argument("--bus", default="101",
                   help="battery bus: RTS-GMLC Bus Name (e.g. Abel) or Bus ID (e.g. 101)")
    b.add_argument("--p-charge-max", type=float, default=50.0,
                   help="max charging power [MW]")
    b.add_argument("--p-discharge-max", type=float, default=50.0,
                   help="max discharging power [MW]")
    b.add_argument("--soc-max", type=float, default=200.0,
                   help="max state of charge / energy capacity [MWh]")
    b.add_argument("--soc-min", type=float, default=0.0,
                   help="min state of charge [MWh]")
    b.add_argument("--soc-init", type=float, default=None,
                   help="initial state of charge [MWh] (default: --soc-min)")
    b.add_argument("--eta-charge", type=float, default=1.0,
                   help="charging efficiency")
    b.add_argument("--eta-discharge", type=float, default=1.0,
                   help="discharging efficiency")
    b.add_argument("--buy-price", type=float, default=10.0,
                   help="charge when DA LMP < this [$/MWh]")
    b.add_argument("--sell-price", type=float, default=15.0,
                   help="discharge when DA LMP > this [$/MWh]")
    return p.parse_args(argv)


def build_options(args):
    job_name = args.job_name or f"battery_{args.bus}"
    output_folder = os.path.join(this_file_path, "results")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{job_name}_results")

    data_path = os.path.abspath(args.data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"RTS-GMLC SourceData folder not found: {data_path}")

    print(f"data:   {data_path}")
    print(f"output: {output_path}")

    shortfall = 500

    battery = {
        "module": PLUGIN_PATH,
        "battery_bus": args.bus,
        "battery_p_charge_max": args.p_charge_max,
        "battery_p_discharge_max": args.p_discharge_max,
        "battery_soc_max": args.soc_max,
        "battery_soc_min": args.soc_min,
        "battery_charge_efficiency": args.eta_charge,
        "battery_discharge_efficiency": args.eta_discharge,
        "battery_buy_price": args.buy_price,
        "battery_sell_price": args.sell_price,
    }
    if args.soc_init is not None:
        battery["battery_soc_init"] = args.soc_init

    prescient_options = {
        "data_path": data_path,
        "reserve_factor": 0.1,
        "simulate_out_of_sample": True,
        "output_directory": output_path,
        "monitor_all_contingencies": False,
        "input_format": "rts-gmlc",
        "start_date": args.start_date,
        "num_days": args.num_days,
        "sced_horizon": 1,
        "ruc_mipgap": 0.01,
        "deterministic_ruc_solver": args.solver,
        "sced_solver": args.solver,
        "sced_frequency_minutes": 60,
        "ruc_horizon": 36,
        # required: the battery reads the DA LMPs from the DA pricing run
        "compute_market_settlements": True,
        "output_solver_logs": False,
        "price_threshold": shortfall,
        "transmission_price_threshold": None,
        "contingency_price_threshold": None,
        "reserve_price_threshold": None,
        "day_ahead_pricing": "aCHP",
        "enforce_sced_shutdown_ramprate": False,
        "ruc_slack_type": "ref-bus-and-branches",
        "sced_slack_type": "ref-bus-and-branches",
        "disable_stackgraphs": True,
        "symbolic_solver_labels": True,
        "output_ruc_solutions": False,
        "write_deterministic_ruc_instances": False,
        "write_sced_instances": False,
        "print_sced": False,
        "plugin": {"battery": battery},
    }
    if args.solver == "gurobi":
        prescient_options["sced_solver_options"] = {"threads": 1}

    return prescient_options


if __name__ == "__main__":
    Prescient().simulate(**build_options(parse_args()))
