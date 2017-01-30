#!/bin/bash -l
#:
#PBS -N val_nets
#PBS -l ngpus=1
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=4:00:00
#PBS -l gputype=K40

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
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models'
  val_script=$working_dir'/validate_all_snapshots.py'
  vis_script=$working_dir'/vis_finetune.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models'
  val_script=$working_dir'/validate_all_snapshots.py'
  vis_script=$working_dir'/vis_finetune.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi

network_='cstrip-fcn32s-color'
snapshot_=''
log_filename=$working_dir'/'$network_'/logs/trainval_acc_vis.log'
python $val_script --test_type 'trainval' --network_dir $network_ >> $log_filename 2>&1
# python $vis_script $log_filename
echo 'logfilename '$log_filename
