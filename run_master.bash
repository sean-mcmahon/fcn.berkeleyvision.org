#!/bin/bash -l
#PBS -N master
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=86:00:00

module load python/2.7.11-foss-2016a

hpc_dir='/home/n8307628'
working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_paramsearch/'
python_script=$working_dir'master.py'
# Because using MKL Blas on HPC
export MKL_CBWR=AUTO

python $python_script
