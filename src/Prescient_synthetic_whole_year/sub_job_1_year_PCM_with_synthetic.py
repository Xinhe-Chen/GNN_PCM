"""Submit a whole-year Prescient PCM run on a synthetic RTS-GMLC case to the cluster.

The case is the "{suffix}_rts_gmlc" folder built by
generate_synthetic_rts_gmlc_whole_year.py; the same suffix is handed to
pcm_run_using_synthetic.py, which writes to results/{suffix}_results.

Usage
-----
    python sub_job_1_year_PCM_with_synthetic.py                 # submit case "test"
    python sub_job_1_year_PCM_with_synthetic.py case01          # submit case "case01"
    python sub_job_1_year_PCM_with_synthetic.py case01 --dry-run  # write the script only
"""

import argparse
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))

RUN_SCRIPT = "pcm_run_using_synthetic.py"
CONDA_ENV = "PCM0826"
EMAIL = "xchen24@nd.edu"
QUEUE = "long"


def write_job_script(suffix, job_name, queue=QUEUE, email=EMAIL, env=CONDA_ENV):
    """Write the SGE submission script for one whole-year run, and return its path."""
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    os.makedirs(job_scripts_dir, exist_ok=True)

    os.makedirs(os.path.join(this_file_path, "sim_job_logs"), exist_ok=True)

    file_name = os.path.join(job_scripts_dir, f"{job_name}.sh")

    # Paths inside the script stay relative: it may be written on one machine and
    # run on the cluster, so no absolute path from here would resolve there. The
    # job cds to the script's own parent directory, which is this package.
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
            "cd \"$(dirname \"$(readlink -f \"$0\")\")/..\"\n"
            "\n"
            # a batch shell is not interactive, so conda's shell function has to be
            # sourced before `conda activate` works
            'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
            f"conda activate {env}\n"
            "module load gurobi\n"
            "\n"
            f"python {RUN_SCRIPT} {suffix}\n"
        )

    return file_name


def submit_job(suffix, job_name=None, queue=QUEUE, email=EMAIL, env=CONDA_ENV,
               dry_run=False):
    if job_name is None:
        job_name = f"{suffix}_1_year_PCM"

    case_path = os.path.join(this_file_path, f"{suffix}_rts_gmlc")
    if not os.path.isdir(os.path.join(case_path, "SourceData")):
        raise FileNotFoundError(
            f"synthetic case not found: {case_path}\n"
            f"build it first with:\n"
            f"    python generate_synthetic_rts_gmlc_whole_year.py --suffix {suffix}"
        )

    file_name = write_job_script(suffix, job_name, queue=queue, email=email, env=env)
    print(f"case:        {case_path}")
    print(f"job script:  {file_name}")

    if dry_run:
        print("\n--dry-run: script written, not submitted. Contents:\n")
        with open(file_name) as f:
            print(f.read())
        return file_name

    print(f"submitting:  qsub {file_name}")
    status = os.system(f"qsub {file_name}")
    if status != 0:
        raise RuntimeError(f"qsub failed with exit status {status}")
    return file_name


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Submit a whole-year synthetic-case PCM run to the cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("suffix", nargs="?", default="test",
                   help="the synthetic case to run ('{suffix}_rts_gmlc')")
    p.add_argument("--job-name", default=None,
                   help="SGE job name (default: '{suffix}_1_year_PCM')")
    p.add_argument("--queue", default=QUEUE, help="SGE queue")
    p.add_argument("--email", default=EMAIL, help="address for job mail")
    p.add_argument("--env", default=CONDA_ENV, help="conda environment to activate")
    p.add_argument("--dry-run", action="store_true",
                   help="write the job script and print it, without calling qsub")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    submit_job(
        suffix=args.suffix,
        job_name=args.job_name,
        queue=args.queue,
        email=args.email,
        env=args.env,
        dry_run=args.dry_run,
    )
