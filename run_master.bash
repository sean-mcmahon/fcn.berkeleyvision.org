#!/bin/bash -l
#PBS -N master
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=48:00:00

module load python/2.7.11-foss-2016a

hpc_dir='/home/n8307628/Fully-Conv-Network/Resources/FCN_paramsearch/'
local_dir='/home/sean/Dropbox/Uni/Code/FCN_models/'
dir_=$local_dir

python_script=$dir_'master.py'
# Because using MKL Blas on HPC
export MKL_CBWR=AUTO

solver_name='solve_any_live'
worker_name='worker_live'
cp $dir_'worker.bash' $dir_$worker_name'.bash'
sed -i -e 's/solve_any/'$solver_name'/g' $dir_$worker_name'.bash'
cp $dir_'solve_any.py' $dir_$solver_name'.py'

python $python_script --worker_name $worker_name
# rm $dir_$solver_name'.py'
