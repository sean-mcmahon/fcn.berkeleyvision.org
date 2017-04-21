#!/bin/bash -l
#PBS -N master
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=48:00:00

module load python/2.7.11-foss-2016a

hpc_dir='/home/n8307628/Fully-Conv-Network/Resources/FCN_paramsearch/'
local_dir='/home/sean/Dropbox/Uni/Code/FCN_models/'
working_dir=$hpc_dir

python_script=$working_dir'master.py'
# Because using MKL Blas on HPC
export MKL_CBWR=AUTO

cp $working_dir'worker.bash' $working_dir'worker_live.bash'
sed -i -e 's/solve_any/solve_any_live/g' $working_dir'worker_live.bash'
cp $working_dir'solve_any.py' $working_dir'solve_any_live.py'

python $python_script
