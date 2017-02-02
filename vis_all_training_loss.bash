#!/bin/bash -l

hpc_dir='/home/n8307628'
local_dir='/home/sean'
if [[ -d $local_dir ]]; then
  working_dir=$local_dir'/hpc-home/Fully-Conv-Network/Resources/FCN_models'
  vis_script=$working_dir'/vis_finetune.py'
elif [[ -d $hpc_dir ]]; then
  working_dir=$hpc_dir'/Fully-Conv-Network/Resources/FCN_models'
  vis_script=$working_dir'/vis_finetune.py'
else
  echo "No directory found..."
fi

color_logpath=$working_dir'/cstrip-fcn32s-color/logs/FCNcolor_train2016-08-22_12-20-47.log'
colorD_logpath=$working_dir'/cstrip-fcn32s-color-d/logs/FCNcolorDepth_train2016-08-25_10-43-23.log'
colorD2_rgbPre_logpath=$working_dir'/cstrip-fcn32s-color-d/logs/FCNcolorDepth_color_init_2nd_2017-01-31_12-39-37.log'
colorD2_rgbdPre_logpath=$working_dir'/cstrip-fcn32s-color-d/logs/FCNcolorDepth_color_pretrainDepth_init_2nd_2017-01-31_12-34-33.log'
colorD2_rgbdPre_lowerLR_logpath=$working_dir'/cstrip-fcn32s-color-d/logs/FCNcolorDepth_colordepth_init_lower_2e10LR_2nd_2017-01-31_13-00-13.log'
depth_logpath=$working_dir'/cstrip-fcn32s-depth/logs/FCNdepth_train2017-01-23_13-23-32_negOneNull.log'
hha_logpath=$working_dir'/cstrip-fcn32s-hha/logs/FCNhha_train2016-08-30_21-50-25.log'
colorHHA_logpath=$working_dir'/cstrip-fcn32s-color-hha-early/logs/FCNcolorHhaEarly_train2017-01-23_16-31-20_pretrainHha.log'

# python $vis_script $color_logpath $colorD_logpath $depth_logpath $hha_logpath $colorHHA_logpath

python $vis_script $colorD2_rgbPre_logpath $colorD2_rgbdPre_logpath $colorD2_rgbdPre_lowerLR_logpath $colorHHA_logpath
