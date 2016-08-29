#!/bin/bash -l
#:
#PBS -N FCN_colorHHA
#PBS -l ncpus=2
#PBS -l mem=64GB
#PBS -l walltime=95:00:00

# module load python
# module load caffe
# module load cuda

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
    for process in $PROCESSES; do
        echo $process
        if [[ "$process" == "NoneHereBro" ]]; then
            GPU_ID_One=$ind
            # break
        fi
        ind=$[ind + 1]
    done
else
    # Not on HPC (probably)
    GPU_ID_One=$(nvidia-smi --list-gpus | awk '{NR;}END {print NR-1}') # Grabs number of GPUS
fi

if [ $USEGPU == 'true' ]; then
    echo "Using gpu: $GPU_ID_One"
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
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi
solver_script=$working_dir'/cstrip-fcn32s-color-hha/solver.prototxt'
pretrained_weights=$working_dir'/pretrained_weights/nyud-fcn32s-ColorHHA_iter_0.caffemodel'

current_date=`date +%Y-%m-%d_%H-%M-%S`
log_filename=$working_dir'/logs/FCNcolorHHA_train'$current_date'.log'

# ./../../../../../pkg/suse11/caffe/20150420/bin/caffe.bin train -solver $solver_script -weights $pretrained_weights -gpu $GPU_ID_One 2>&1 | tee $log_filename
echo $log_filename
