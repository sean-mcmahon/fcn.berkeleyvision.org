#!/bin/bash -l
#PBS -N rgbhha_c_4b
#PBS -l ngpus=1
#PBS -l ncpus=1
#PBS -l mem=32GB
#PBS -l walltime=5:00:00
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
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_paramsearch/'
  python_script=$working_dir'solve_any.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_paramsearch/'
  python_script=$working_dir'solve_any.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi


current_date=`date +%Y-%m-%d_%H-%M-%S`
train_folder_="$1"
if [[ -z "$train_folder_" ]]; then
  if [ -z ${MY_TRAIN_DIR+x} ]; then
    train_folder_=$working_dir'worker_test_'$current_date;
  else train_folder_=$MY_TRAIN_DIR
  fi
fi
cv_fold_idx="$2"
if [[ -z "$cv_fold_idx" ]]; then
  if [ -z ${MY_CV_FOLD+x} ]; then
    cv_fold_idx='o';
  else cv_fold_idx=$MY_CV_FOLD
  fi
fi

set_mode="$3"
if [[ -z "$set_mode" ]]; then
  set_mode='gpu'
fi

if [ -z ${MY_BASELR+x} ]; then
  base_lr_=
else base_lr_=$MY_BASELR
fi
echo "train_folder_="$train_folder_
mkdir -p $train_folder_
echo "cv_fold_idx="$MY_CV_FOLD

jobID=$PBS_JOBID
echo 'Job ID: '$PBS_JOBID
echo 'Job ID: '$PBS_JOBID >> $train_folder_'/'$PBS_JOBID'.txt'

log_filename=$train_folder_'/logfile'$current_date'.log'
echo 'log_filename '$log_filename
python $python_script --mode $set_mode --working_dir $train_folder_ --traintest_fold $cv_fold_idx 2>&1 | tee $log_filename
echo 'Saved to '$log_filename
