#!/bin/bash
#SBATCH -J BOSS_FinePlot # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one Node
#SBATCH -p shared # Partition
#SBATCH --mem 4000 # Memory request (4Gb)
#SBATCH -t 0-2:00 # Maximum execution time (D-HH:MM)
#SBATCH -o "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_%a.out" # Standard output
#SBATCH -e "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_%a.err" # Standard error
#SBATCH --account=dvorkin_lab
#SBATCH --mail-type=ALL        ## BEGIN, END, FAIL, ALL                         
#SBATCH --mail-user=nicholasdeporzio@g.harvard.edu 

## SETUP STORAGE ##                                                             
PROJECT_DIR="${HOME}/projects/cosmicfish/cfworkspace/"                          
STORAGE_DIR="${HOME}/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}/" 
DATASTORE_DIR="${SCRATCH}/dvorkin_lab/ndeporzio/cosmicfish/FinePlotData/data_FinePlot_${SLURM_ARRAY_TASK_ID}/"           
CLASS_DIR="/n/dvorkin_lab/ndeporzio/class/boss_fine/class${SLURM_ARRAY_TASK_ID}/"        
PRIORS_DIR="/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/cosmicfish/priors/"
FORECAST_INDEX="${SLURM_ARRAY_TASK_ID}"

mkdir -p $STORAGE_DIR
mkdir -p $DATASTORE_DIR

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
echo "FORECAST_INDEX : ${SLURM_ARRAY_TASK_ID}"

export PROJECT_DIR                                                              
export STORAGE_DIR                                                              
export DATASTORE_DIR                                                            
export CLASS_DIR                                                                
export PRIORS_DIR                                                               
export SLURM_STEP_NUM_TASKS  
export FORECAST_INDEX                                                   
 
## COPY INPUT FILES AND MOVE TO WORKING DIR ##                                  
cd /n/home02/ndeporzio/projects/cosmicfish/ 

## LOAD REQUIRED MODULES/ENVIRONMENT ##                                         
module load Anaconda/5.0.1-fasrc02                                              
source activate /n/home02/ndeporzio/projects/cosmicfish/py36 

## RUN PROGRAM ##                                                               
SCRIPT_DIR="/n/home02/ndeporzio/projects/cosmicfish/cosmicfish/bin/boss_fine_plot_job_array.py"
cp $SCRIPT_DIR $STORAGE_DIR                                                     
python $SCRIPT_DIR >> "${HOME}/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.scriptout" 

## COPY OUTPUT TO SECURE STORAGE ##
if [[ -f "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.out" ]]
then
    mv "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.out" "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.out"
fi

if [[ -f "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.err" ]]
then
    mv "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.err" "/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS_18SEP2020/BOSS/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}/BOSS_FinePlot_${SLURM_ARRAY_TASK_ID}.err"
fi
