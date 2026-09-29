"""Submit a whole-year Prescient PCM run with the rule-based battery to the cluster.

The job runs battery_pcm_run.py, which writes to results/{job_name}_results
(the usual Prescient outputs plus battery_results.csv).

Battery and case options are passed straight through to battery_pcm_run.py.
Only the options you give here are forwarded; everything else uses the
defaults in battery_pcm_run.py (origin RTS-GMLC case, 01-01-2020, 366 days,
50 MW / 50 MW / 200 MWh battery at bus 101, buy < 10, sell > 15 $/MWh).

Usage
-----
    python sub_job_1_year_battery_PCM.py                                  # all defaults
    python sub_job_1_year_battery_PCM.py --bus 313 --p-charge-max 100 \
        --p-discharge-max 100 --soc-max 400                               # another battery
    python sub_job_1_year_battery_PCM.py --bus Abel --dry-run             # write the script only

A relative --data-path is resolved from this folder (src/battery_PCM), since
the job runs here.
"""

import argparse
import os
import shlex
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))

RUN_SCRIPT = "battery_pcm_run.py"
CONDA_ENV = "PCM0826"
EMAIL = "xchen24@nd.edu"
QUEUE = "long"

# options forwarded to battery_pcm_run.py: (sub_job dest, run-script flag)
FORWARDED = [
    ("data_path", "--data-path"),
    ("start_date", "--start-date"),
    ("num_days", "--num-days"),
    ("solver", "--solver"),
    ("bus", "--bus"),
    ("p_charge_max", "--p-charge-max"),
    ("p_discharge_max", "--p-discharge-max"),
    ("soc_max", "--soc-max"),
    ("soc_min", "--soc-min"),
    ("soc_init", "--soc-init"),
    ("eta_charge", "--eta-charge"),
    ("eta_discharge", "--eta-discharge"),
    ("buy_price", "--buy-price"),
    ("sell_price", "--sell-price"),
]


def build_run_command(args, job_name):
    """The python command the job runs, with only the options that were given."""
    cmd = ["python", RUN_SCRIPT, "--job-name", job_name]
    for dest, flag in FORWARDED:
        value = getattr(args, dest)
        if value is not None:
            cmd += [flag, str(value)]
    return " ".join(shlex.quote(c) for c in cmd)


def write_job_script(job_name, run_command, queue=QUEUE, email=EMAIL, env=CONDA_ENV):
    """Write the SGE submission script for one whole-year run, and return its path."""
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    os.makedirs(job_scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(this_file_path, "sim_job_logs"), exist_ok=True)

    file_name = os.path.join(job_scripts_dir, f"{job_name}.sh")

    # Paths inside the script stay relative: it may be written on one machine and
    # run on the cluster. The job is submitted from this folder (see submit_job),
    # so "#$ -cwd" makes this folder the working directory.
    with open(file_name, "w", newline="\n") as f:
        f.write(
            "#!/bin/bash\n"
            f"#$ -M {email}\n"
            "#$ -m ae\n"
            f"#$ -q {queue}\n"
            f"#$ -N {job_name}\n"
            "#$ -cwd\n"
            f"#$ -o sim_job_logs/{job_name}.out\n"
            f"#$ -e sim_job_logs/{job_name}.err\n"
            "\n"
            "set -e\n"
            # a batch shell is not interactive, so conda's shell function has to be
            # sourced before `conda activate` works
            'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
            f"conda activate {env}\n"
            "module load gurobi\n"
            "\n"
            f"{run_command}\n"
        )

    return file_name


def submit_job(args):
    job_name = args.job_name or f"battery_{args.bus or '101'}_1_year_PCM"

    if args.data_path is not None:
        data_path = os.path.join(this_file_path, args.data_path)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"RTS-GMLC SourceData folder not found: {data_path}")

    run_command = build_run_command(args, job_name)
    file_name = write_job_script(job_name, run_command,
                                 queue=args.queue, email=args.email, env=args.env)
    print(f"job script:  {file_name}")
    print(f"command:     {run_command}")

    if args.dry_run:
        print("\n--dry-run: script written, not submitted. Contents:\n")
        with open(file_name) as f:
            print(f.read())
        return file_name

    # submit from this folder so "#$ -cwd" (and the relative paths) point here
    print(f"submitting:  qsub {file_name}")
    result = subprocess.run(["qsub", file_name], cwd=this_file_path)
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed with exit status {result.returncode}")
    return file_name


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Submit a whole-year battery PCM run to the cluster. Case and "
                    "battery options left unset use battery_pcm_run.py's defaults.",
    )
    # --- job ---
    j = p.add_argument_group("job")
    j.add_argument("--job-name", default=None,
                   help="SGE job name and output label (default: 'battery_{bus}_1_year_PCM')")
    j.add_argument("--queue", default=QUEUE, help=f"SGE queue (default: {QUEUE})")
    j.add_argument("--email", default=EMAIL, help="address for job mail")
    j.add_argument("--env", default=CONDA_ENV,
                   help=f"conda environment to activate (default: {CONDA_ENV})")
    j.add_argument("--dry-run", action="store_true",
                   help="write the job script and print it, without calling qsub")

    # --- case / simulation window ---
    c = p.add_argument_group("case")
    c.add_argument("--data-path", default=None,
                   help="RTS-GMLC SourceData folder, relative to src/battery_PCM or absolute")
    c.add_argument("--start-date", default=None, help="MM-DD-YYYY")
    c.add_argument("--num-days", type=int, default=None)
    c.add_argument("--solver", default=None)

    # --- battery ---
    b = p.add_argument_group("battery")
    b.add_argument("--bus", default=None, help="Bus Name (e.g. Abel) or Bus ID (e.g. 101)")
    b.add_argument("--p-charge-max", type=float, default=None, help="[MW]")
    b.add_argument("--p-discharge-max", type=float, default=None, help="[MW]")
    b.add_argument("--soc-max", type=float, default=None, help="[MWh]")
    b.add_argument("--soc-min", type=float, default=None, help="[MWh]")
    b.add_argument("--soc-init", type=float, default=None, help="[MWh]")
    b.add_argument("--eta-charge", type=float, default=None)
    b.add_argument("--eta-discharge", type=float, default=None)
    b.add_argument("--buy-price", type=float, default=None, help="[$/MWh]")
    b.add_argument("--sell-price", type=float, default=None, help="[$/MWh]")
    return p.parse_args(argv)


if __name__ == "__main__":
    submit_job(parse_args())
