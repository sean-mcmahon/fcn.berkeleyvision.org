#!/bin/bash -l

module load python/2.7.11-foss-2016a
hpc_dir='/home/n8307628'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_paramsearch/'
  python_script=$working_dir'/run_crossval.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_paramsearch/'
  python_script=$working_dir'run_crossval.py'
  # Because using MKL Blas on HPC
  export MKL_CBWR=AUTO
else
  echo "No directory found..."
fi

python $python_script
