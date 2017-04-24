#!/bin/bash -l
#PBS -N master_depth
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=28:00:00

module load python/2.7.11-foss-2016a

hpc_dir='/home/n8307628/Fully-Conv-Network/Resources/FCN_paramsearch/'
local_dir='/home/sean/Dropbox/Uni/Code/FCN_models/'
dir_=$hpc_dir

python_script=$dir_'master.py'
# Because using MKL Blas on HPC
export MKL_CBWR=AUTO

solver_name='solve_any_live_depth'
worker_name='worker_live_depth'
cp $dir_'worker.bash' $dir_$worker_name'.bash'
sed -i -e 's/solve_any/'$solver_name'/g' $dir_$worker_name'.bash'
cp $dir_'solve_any.py' $dir_$solver_name'.py'

data_type='depth'
sess_=$data_type'_workers'
workers_=$data_type'_1_'
job_time_=24
num_wrks_=2

python $python_script --worker_name $worker_name --session_dir $sess_ --worker_id_dir $workers_ --run_time $job_time_ --max_workers $num_wrks_
mv $dir_$solver_name'.py' $dir_$sess_'/'
mv $dir_$worker_name'.bash' $dir_$sess_'/'
