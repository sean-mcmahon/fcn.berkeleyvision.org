#!/bin/bash -l
#:
#PBS -N val_nets
#PBS -l ngpus=0
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l walltime=4:00:00

module load python/2.7.11-foss-2016a

hpc_dir='/home/n8307628'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models'
  vis_script=$working_dir'/vis_finetune.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models'
  vis_script=$working_dir'/vis_finetune.py'
  # Because using MKL Blas on HPC
else
  echo "No directory found..."
fi

network_='cstrip-fcn32s-color'
log_filename=$working_dir'/'$network_'/logs/trainval_acc_vis.log'
python $vis_script $log_filename
echo 'logfilename '$log_filename

network_='cstrip-fcn32s-depth'
snapshot_='negOneNull_mean_sub'
log_filename=$working_dir'/'$network_'/logs/trainval_acc_vis_'$snapshot_'.log'
python $vis_script $log_filename
echo 'logfilename '$log_filename

network_='cstrip-fcn32s-color-d'
snapshot_='colorInit_5xLR'
log_filename=$working_dir'/'$network_'/logs/trainval_acc_vis_'$snapshot_'.log'
python $vis_script $log_filename
echo 'logfilename '$log_filename

network_='cstrip-fcn32s-hha'
snapshot_='secondTrain_lowerLR'
log_filename=$working_dir'/'$network_'/logs/trainval_hha_vis_'$snapshot_'.log'
python $vis_script $log_filename
echo 'logfilename '$log_filename

network_='cstrip-fcn32s-color-hha-early'
snapshot_='colorHhaInit_5xLR'
log_filename=$working_dir'/'$network_'/logs/trainval_rgbhha_vis_'$snapshot_'.log'
python $vis_script $log_filename
echo 'logfilename '$log_filename
