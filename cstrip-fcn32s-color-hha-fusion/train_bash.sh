#!/bin/bash -l
#:
#PBS -N FCN_colorHHA
#PBS -l ncpus=2
#PBS -l mem=64GB
#PBS -l walltime=95:00:00

module load python
module load caffe
module load cuda

USEGPU='true'
if [[ $(lsb_release -si) == *"SUSE LINUX"* ]]; then
    # On HPC (probably)

    # Old GPU ID method only works on nodes with 2x GPUs
    # GPU_ID=$(nvidia-smi | awk '{ if(NR==19) if($2>0) print 0; else print 1 }')

    # New GPU ID method works on nodes with 1 or more GPUs
    PROCESSES=$((nvidia-smi -q -d pids | grep Processes) | awk '{printf "%sHereBro ",$3}')
    ind=0
    GPU_ID=-1
    for process in $PROCESSES; do
        echo $process
        if [[ "$process" == "NoneHereBro" ]]; then
            GPU_ID=$ind
            break
        fi
        ind=$[ind + 1]
    done
else
    # Not on HPC (probably)
    GPU_ID=$(nvidia-smi --list-gpus | awk '{NR;}END {print NR-1}') # Grabs number of GPUS
fi

if [ $USEGPU == 'true' ]; then
    echo "Using gpu: $GPU_ID"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    gpu=$GPU_ID
else
    echo "Using cpu"
    gpu=-1
fi

hpc_dir='/home/n8307628'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color-hha-fusion'
  python_script=$working_dir'/solve.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color-hha-fusion'
  python_script=$working_dir'/solve.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi

set_mode="$1"
if [[ -z "$set_mode" ]]; then
  set_mode='gpu'
fi
current_date=`date +%Y-%m-%d_%H-%M-%S`
log_filename=$working_dir'/logs/FCNcolorhhaFusion_train'$current_date'.log'

python $python_script --mode $set_mode 2>&1 | tee $log_filename
echo $log_filename
