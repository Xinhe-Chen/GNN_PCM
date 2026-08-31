#!/bin/bash
#$ -M xchen24@nd.edu
#$ -m ae
#$ -q long
#$ -N test_1_year_PCM
#$ -cwd
#$ -o sim_job_logs/test_1_year_PCM.out
#$ -e sim_job_logs/test_1_year_PCM.err

set -e
cd "$(dirname "$(readlink -f "$0")")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PCM0826
module load gurobi

python pcm_run_using_synthetic.py test
