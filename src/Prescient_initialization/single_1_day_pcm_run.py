import os
from prescient.simulator import Prescient

this_file_path = os.path.dirname(os.path.realpath(__file__))

# default some options
shortfall = 500
output_folder = os.path.join(this_file_path, "results")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "test_1_day_PCM")
data_path = os.path.join(this_file_path, "synthetic_gmlc", "RTS_Data", "SourceData")

prescient_options = {
        "data_path":data_path,
        "reserve_factor": 0.1,
        "simulate_out_of_sample":True,
        "output_directory":output_path,
        "monitor_all_contingencies":False,
        "input_format":"rts-gmlc",
        "start_date":"01-01-2020",
        "num_days":1,
        "sced_horizon":1,
        "ruc_mipgap":0.001,
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