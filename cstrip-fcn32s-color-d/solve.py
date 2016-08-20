#! /usr/bin/python
"""
cstrip color DEPTH

"""
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
    weights = home_dir+'/Fully-Conv-Network/Resources/FCN_models/pretrained_weights/nyud-fcn32s-color-heavy.caffemodel'
elif 'sean' in home_dir:
    caffe_root = home_dir+'/src/caffe'
    weights = home_dir+'/hpc-home/Fully-Conv-Network/Resources/FCN_models/pretrained_weights/nyud-fcn32s-color-heavy.caffemodel'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
caffe.set_mode_gpu()
import surgery, score

# init
base_net_arch  = file_location[:file_location.rfind('/')]+'/cstrip-fcn32s-color/test.prototxt'
base_net = caffe.Net(base_net_arch, weights,
        caffe.TEST)
solver = caffe.SGDSolver('solver.prototxt')
surgery.transplant(solver.net, base_net)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)
solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params['conv1_1'][0].data
solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params['conv1_1'][1].data
del base_net

# scoring
val = np.loadtxt(file_location[:file_location.rfind('/')]+'/data/cs-trip/val.txt', dtype=str)

for _ in range(50):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(2000)
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
    # score.seg_tests(solver, False, val, layer='score')
