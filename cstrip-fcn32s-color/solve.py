#! /usr/bin/python
# import caffe
import numpy as np
import os, sys
from os.path import expanduser
import imp

# add '../' directory to path for importing score.py, surgery.py and pycaffe layer
file_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
home_dir = expanduser("~")

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir+'/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir+'/src/caffe'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
caffe.set_mode_cpu()
# caffe.set_device(1)
import surgery, score

# init
weights = file_location+'/nyud-fcn32s-color-heavy.caffemodel'
solver = caffe.SGDSolver(file_location+'/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery'
surgery.interp(solver.net, interp_layers)
#
# # scoring
val = np.loadtxt(home_dir+'/Construction_Site/Springfield/12Aug16/K2/val.txt', dtype=str)

for _ in range(50):
    print 'Running solver.step iter {}'.format(_)
    solver.step(2000)
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    print 'Validation'
    score.seg_tests(solver, False, val, layer='score')
