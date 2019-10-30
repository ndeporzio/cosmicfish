#!/bin/bash

#SBATCH -J relic_science_plot_1 ##Job name
#SBATCH -o relic_science_plot_1.out ##Write stdout to this file
#SBATCH -e relic_science_plot_1.err ##Write error messages to this file
#SBATCH --open-mode=append ##Not sure
#SBATCH -p shared ##Use the 'shared' partition 
#SBATCH -t 2-0:00 ##Run up to 2 days, 0 hours, 0 minutes
#SBATCH -n 8 ##Number of cores 
#SBATCH --mem=6000 ##Use no more than 6Gb RAM 
#SBATCH -N 1 ##Use no more than 1 compute node

## SETUP STORAGE ##
STORAGE_DIR="/n/home02/ndeporzio/projects/cosmicfish/results/${SLURM_JOB_ID}"

echo "STORAGE_DIR: $STORAGE_DIR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_STEP_NUM_TASKS: $SLURM_STEP_NUM_TASKS"

export STORAGE_DIR
export SLURM_STEP_NUM_TASKS
mkdir $STORAGE_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR ##
cd ~/projects/cosmicfish/

## LOAD REQUIRED MODULES/ENVIRONMENT ##
module load centos6/0.0.1-fasrc01
module load virtualenv
module load gcc/4.9.0-fasrc01
source activate ./py36 

## RUN PROGRAM ##
#python "$SLURM_SUBMIT_DIR/relic_science_plot_1.py" >> relic_science_plot_1.out
PROJECT_DIR = "/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/bin/relic_science_plot_1.py"
python $PROJECT_DIR  >> relic_science_plot_1.out
##output=relic_science_plot_1_output.csv

## COPY OUTPUT TO SECURE STORAGE ##
##cp $output $SLURM_SUBMIT_DIR/$output
