import os
import sys
from prescient.simulator import Prescient

this_file_path = os.path.dirname(os.path.realpath(__file__))

# which synthetic case to run: the "{suffix}_rts_gmlc" folder built by
# generate_synthetic_rts_gmlc_whole_year.py. Override from the command line
# ("python pcm_run_using_synthetic.py case01 [job_name]") to run a different case.
suffix = sys.argv[1] if len(sys.argv) > 1 else "test"
# the name of this run, which labels the output folder. Defaults to the case
# suffix, so a second argument only matters when several runs share one case.
job_name = sys.argv[2] if len(sys.argv) > 2 else suffix
case_path = os.path.join(this_file_path, f"{suffix}_rts_gmlc")

# default some options
shortfall = 500
output_folder = os.path.join(this_file_path, "results")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, f"{job_name}_results")

# the rts-gmlc parser reads bus/gen/timeseries_pointers from SourceData and
# resolves the time series files relative to it
data_path = os.path.join(case_path, "SourceData")

if not os.path.isdir(data_path):
    raise FileNotFoundError(
        f"synthetic case not found: {data_path}\n"
        f"build it first with:\n"
        f"    python generate_synthetic_rts_gmlc_whole_year.py --suffix {suffix}"
    )

print(f"case:   {case_path}")
print(f"data:   {data_path}")
print(f"output: {output_path}")

prescient_options = {
        "data_path":data_path,
        "reserve_factor": 0.1,
        "simulate_out_of_sample":True,
        "output_directory":output_path,
        "monitor_all_contingencies":False,
        "input_format":"rts-gmlc",
        "start_date":"01-01-2020",
        "num_days":366,
        "sced_horizon":1,
        "ruc_mipgap":0.01,
	    "deterministic_ruc_solver": "gurobi",
        "sced_solver":"gurobi",
        "sced_frequency_minutes":60,
	    "sced_solver_options" : {"threads":1},
        "ruc_horizon":36,
        "compute_market_settlements":True,
        "output_solver_logs":False,
        "price_threshold":shortfall,
        "transmission_price_threshold":None,
        "contingency_price_threshold":None,
        "reserve_price_threshold":None,
        "day_ahead_pricing":"aCHP",
        "enforce_sced_shutdown_ramprate":False,
        "ruc_slack_type":"ref-bus-and-branches",
        "sced_slack_type":"ref-bus-and-branches",
	    "disable_stackgraphs":True,
        "symbolic_solver_labels":True,
        "output_ruc_solutions": False,
        "write_deterministic_ruc_instances": False,
        "write_sced_instances": False,
        "print_sced":False
        }

Prescient().simulate(**prescient_options)