#!/bin/bash

#SBATCH -J plot_set_17         ##Job name
#SBATCH -o plot_set_17.out     ##Write stdout to this file
#SBATCH -e plot_set_17.err     ##Write error messages to this file
#SBATCH --open-mode=append     ##Not sure
#SBATCH -p shared              ##Use the 'shared' partition 
#SBATCH -t 4-0:00              ##Run up to 4 days, 0 hours, 0 minutes
#SBATCH -n 1                   ##Number of cores 
#SBATCH --mem=15000             ##Use no more than 15Gb RAM 
#SBATCH -N 1                   ##Use no more than 1 compute node
#SBATCH --account=dvorkin_lab
#SBATCH --mail-type=ALL        ## BEGIN, END, FAIL, ALL
#SBATCH --mail-user=nicholasdeporzio@g.harvard.edu
####  #SBATCH --test-only

## SETUP STORAGE ##
PROJECT_DIR="${HOME}/projects/cosmicfish/cfworkspace/"
STORAGE_DIR="${HOME}/projects/cosmicfish/cfworkspace/results/${SLURM_JOB_ID}"
DATASTORE_DIR="${SCRATCH}/dvorkin_lab/ndeporzio/datastore/data17/"
CLASS_DIR="/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/class17/"
PRIORS_DIR="/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/cosmicfish/priors/"

echo "PROJECT_DIR: $PROJECT_DIR"
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

export PROJECT_DIR
export STORAGE_DIR
export DATASTORE_DIR 
export CLASS_DIR
export PRIORS_DIR
export SLURM_STEP_NUM_TASKS
mkdir -p $STORAGE_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR ##
cd /n/home02/ndeporzio/projects/cosmicfish/

## LOAD REQUIRED MODULES/ENVIRONMENT ##
module load Anaconda/5.0.1-fasrc02
source activate /n/home02/ndeporzio/projects/cosmicfish/py36

## RUN PROGRAM ##
SCRIPT_DIR="/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/bin/plot_set_17.py"
cp $SCRIPT_DIR $STORAGE_DIR
python $SCRIPT_DIR >> "${STORAGE_DIR}/plot_set_17.scriptout"

## COPY OUTPUT TO SECURE STORAGE ##
if [[ -f "${SLURM_SUBMIT_DIR}/plot_set_17.out" ]]
then
    mv "${SLURM_SUBMIT_DIR}/plot_set_17.out" $STORAGE_DIR
fi

if [[ -f "${SLURM_SUBMIT_DIR}/plot_set_17.err" ]]
then
    mv "${SLURM_SUBMIT_DIR}/plot_set_17.err" $STORAGE_DIR
fi

if [[ -f "${SLURM_SUBMIT_DIR}/plot_set_17.scriptout" ]]
then
    mv "${SLURM_SUBMIT_DIR}/plot_set_17.scriptout" $STORAGE_DIR
fi
