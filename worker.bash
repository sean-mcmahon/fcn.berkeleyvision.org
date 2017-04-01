#!/bin/bash -l
#:
#PBS -N worker
#PBS -l ngpus=1
#PBS -l ncpus=1
#PBS -l mem=32GB
#PBS -l walltime=4:00:00
#PBS -l gputype=M40

module load python/2.7.11-foss-2016a
module load caffe
module unload caffe
module load cuda/7.5.18-foss-2016a

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
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_paramsearch'
  python_script=$working_dir'/solve_any.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_paramsearch'
  python_script=$working_dir'/solve_any.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi

current_date=`date +%Y-%m-%d_%H-%M-%S`
set_mode="$1"
if [[ -z "$set_mode" ]]; then
  set_mode='gpu'
fi
training_dir_="$2"
if [[ -z "$training_dir_" ]]; then
  training_dir_='rgb_1'$current_date
fi

log_filename=$working_dir$training_dir_$current_date'.log'
mkdir -p $working_dir$training_dir_
python $python_script --mode $set_mode --working_dir $working_dir$training_dir_  2>&1 | tee $log_filename
echo 'Saved to '$log_filename
