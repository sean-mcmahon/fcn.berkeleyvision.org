#!/bin/bash -l
#:
#PBS -N validate_FCN
#PBS -l ngpus=1
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=4:00:00

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
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models'
  python_script=$working_dir'/deploy.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models'
  python_script=$working_dir'/deploy.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi
network='cstrip-fcn32s-color'
set_mode="$1"
if [[ -z "$set_mode" ]]; then
  set_mode='gpu'
fi
split="$2"
if [[ -z "$split" ]]; then
  split='deploy'
fi
snapshot_iter="$3"
if [[ -z "$snapshot_iter" ]]; then
  snapshot_iter='8000'
fi
save_directory="$4"
if [[ -z "$save_directory" ]]; then
  save_directory='/home/n8307628/Construction_Site/Springfield/12Aug16/K2/2016-08-12-10-09-26_groundfloorCarPark/video_frames/colour_predictions_two'
fi
# current_date=`date +%Y-%m-%d_%H-%M-%S`
log_filename=$working_dir'/'$network'/logs/'$split'_dataset_snapshot_'$snapshot_iter'_deploy.log'

python $python_script --mode $set_mode --test_type $split --iteration $snapshot_iter --network_dir $network --save_dir $save_directory 2>&1 | tee $log_filename
echo "Log saved to:"
echo $log_filename