#! /usr/bin/python
# import caffe
import numpy as np
import os, sys
import imp
# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

# add '../' directory to path for importing score.py, surgery.py and pycaffe layer
file_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])

# import support functions
caffe_root = '/home/n8307628/Fully-Conv-Network/Resources/caffe'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
caffe.set_mode_cpu()
# caffe.set_device(1)
# with caffe found load remaining files (they use caffe too)


weights = file_location+'/nyud-fcn32s-color-heavy.caffemodel'

# init
solver = caffe.SGDSolver(file_location+'/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
import surgery, score
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery'
surgery.interp(solver.net, interp_layers)
#
# # scoring
val = np.loadtxt('/work/cyphy/Datasets/NYU2/val.txt', dtype=str)

for _ in range(50):
    print 'Running solver.step iter {}'.format(_)
    solver.step(2000)
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    print 'Validation'
    score.seg_tests(solver, False, val, layer='score')
