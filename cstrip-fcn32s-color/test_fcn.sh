#!/bin/bash -l
#:
#PBS -N validate_FCN
#PBS -l ngpus=1
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=4:00:00

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
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models/'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models/'

  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi
python_script=$working_dir'/validate.py'

set_mode="$1"
if [[ -z "$set_mode" ]]; then
  set_mode='gpu'
fi
split="$2"
if [[ -z "$split" ]]; then
  split='test'
fi
snapshot_iter="$3"
if [[ -z "$snapshot_iter" ]]; then
  snapshot_iter='6000'
fi
snapshot_filter_="$4"
if [[ -z "$snapshot_filter_" ]]; then
  snapshot_filter_=""
fi
network_dir_="$5"
if [[ -z "$network_dir_" ]]; then
  network_dir_='cstrip-fcn32s-color'
fi
# current_date=`date +%Y-%m-%d_%H-%M-%S`
log_filename=$working_dir'/'$network_dir_'/logs/'$split'_dataset_snapshot_'$snapshot_filter_'_'$snapshot_iter'_results.log'

python $python_script --mode $set_mode --test_type $split --iteration $snapshot_iter --snapshot_filter=$snapshot_filter_ --network_dir $network_dir_ 2>&1 | tee $log_filename
echo "Network cstrip-fcn32s-color, iter "$snapshot_iter
