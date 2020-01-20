#!/bin/bash

#SBATCH -J plot_set_9         ##Job name
#SBATCH -o plot_set_9.out     ##Write stdout to this file
#SBATCH -e plot_set_9.err     ##Write error messages to this file
#SBATCH --open-mode=append              ##Not sure
#SBATCH -p shared               ##Use the 'shared' partition 
#SBATCH -t 2-0:00                       ##Run up to 2 days, 0 hours, 0 minutes
#SBATCH -n 1                            ##Number of cores 
#SBATCH --mem=2000                      ##Use no more than 6Gb RAM 
#SBATCH -N 1                            ##Use no more than 1 compute node
#SBATCH --account=dvorkin_lab
#SBATCH --mail-type=ALL               ## BEGIN, END, FAIL, ALL
#SBATCH --mail-user=nicholasdeporzio@g.harvard.edu
####  #SBATCH --test-only

## SETUP STORAGE ##
STORAGE_DIR="/n/home02/ndeporzio/projects/cosmicfish/results/${SLURM_JOB_ID}"
DATASTORE_DIR="/n/home02/ndeporzio/projects/cosmicfish/datastore/"
CLASS_DIR="/n/home02/ndeporzio/projects/cosmicfish/class/"
PRIORS_DIR="/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/cosmicfish/priors/"

echo "STORAGE_DIR: $STORAGE_DIR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_STEP_NUM_TASKS: $SLURM_STEP_NUM_TASKS"
echo "SLURM_SUBMIT_DIR : $SLURM_SUBMIT_DIR"

export STORAGE_DIR
export DATASTORE_DIR 
export CLASS_DIR
export PRIORS_DIR
export SLURM_STEP_NUM_TASKS
mkdir -p $STORAGE_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR ##
cd /n/home02/ndeporzio/projects/cosmicfish/

## LOAD REQUIRED MODULES/ENVIRONMENT ##
module load centos6/0.0.1-fasrc01
module load virtualenv
module load gcc/4.9.0-fasrc01
source activate ./py36 

## RUN PROGRAM ##
PROJECT_DIR="/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/bin/plot_set_9.py"
OUTPUT_FILE="${STORAGE_DIR}/plot_set_9.py"
python $PROJECT_DIR >> $OUTPUT_FILE 

## COPY OUTPUT TO SECURE STORAGE ##
##cp $output $SLURM_SUBMIT_DIR/$output
