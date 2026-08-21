import os

this_file_path = os.path.dirname(os.path.realpath(__file__))

def submit_job(job_name):
    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"{job_name}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N {job_name}\n"
            + f"conda activate PCM0826\n"
            + "module load gurobi\n"
            + f"python ./single_1_day_pcm_run.py"
        )

    os.system(f"qsub {file_name}")

if __name__ == "__main__":
    job_name = "testing_1_day_PCM"
    submit_job(job_name)