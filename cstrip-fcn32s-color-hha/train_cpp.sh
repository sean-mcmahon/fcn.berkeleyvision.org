#!/bin/bash -l
#:
#PBS -N FCN_colorHHA
#PBS -l ncpus=1
#PBS -l ngpus=2
#PBS -l mem=16GB
#PBS -l walltime=24:00:00
#PBS -l gputype=K40

module load python
module load caffe
module load cuda

USEGPU='true'
if [[ $(lsb_release -si) == *"SUSE LINUX"* ]]; then
    # On HPC (probably)

    # Old GPU ID method only works on nodes with 2x GPUs
    # GPU_ID_One=$(nvidia-smi | awk '{ if(NR==19) if($2>0) print 0; else print 1 }')

    # New GPU ID method works on nodes with 1 or more GPUs
    PROCESSES=$((nvidia-smi -q -d pids | grep Processes) | awk '{printf "%sHereBro ",$3}')
    ind=0
    GPU_ID_One=-1
    GPU_ID_Two=-1
    negative_one=-1
    for process in $PROCESSES; do
        echo $process
        if [[ "$process" == "NoneHereBro" ]]; then
            if [ "$GPU_ID_One" -eq "$negative_one" ]; then
              GPU_ID_One=$ind
            elif [ "$GPU_ID_Two" -eq "$negative_one" ]; then
              GPU_ID_Two=$ind
              break
            fi
        fi
        ind=$[ind + 1]
    done
else
    # Not on HPC (probably)
    GPU_ID_One=$(nvidia-smi --list-gpus | awk '{NR;}END {print NR-1}') # Grabs number of GPUS
fi

if [ $USEGPU == 'true' ]; then
    echo "Using gpu: $GPU_ID_One and: $GPU_ID_Two"
    export CUDA_VISIBLE_DEVICES=$GPU_ID_One
    gpu=$GPU_ID_One
else
    echo "Using cpu"
    gpu=-1
fi

hpc_dir='/home/n8307628'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models'
  caffe_bin=$local_dir'/src/caffe/build/tools/caffe'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models'
  caffe_bin=$hpc_dir'/Fully-Conv-Network/Resources/caffe/build/tools/caffe'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi
solver_script=$working_dir'/cstrip-fcn32s-color-hha/solver.prototxt'
pretrained_weights=$working_dir'/pretrained_weights/nyud-fcn32s-ColorHHA_iter_0.caffemodel'
export PYTHONPATH=$PYTHONPATH:$working_dir

current_date=`date +%Y-%m-%d_%H-%M-%S`
log_filename=$working_dir'/cstrip-fcn32s-color-hha/logs/FCNcolorHHA_train'$current_date'.log'

/./$caffe_bin train -solver $solver_script -weights $pretrained_weights --gpu=$GPU_ID_One","$GPU_ID_Two 2>&1 | tee $log_filename
echo $log_filename
