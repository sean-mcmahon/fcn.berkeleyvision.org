#! /usr/bin/python
"""
cstrip Color and HHA2 Eltwise late fusion

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='GPU')
args = parser.parse_args()
print 'This is the colour-HHA2 summation solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir+'/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir+'/src/caffe'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
if 'g' in args.mode or 'G' in args.mode:
    caffe.set_mode_gpu()
    print '-- GPU Mode -- {}'.format(args.mode)
elif 'c' in args.mode or 'C' in args.mode:
    caffe.set_mode_cpu()
    print '-- CPU Mode -- {}'.format(args.mode)
else:
    caffe.set_mode_gpu()
    print '==============='
    print 'No Mode (CPU or GPU) Given'
    print '-- GPU Mode Chosen --'
    print '==============='
# caffe.set_device(1)
import surgery, score


color_weights = file_parent_dir + \
    '/cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel'
color_proto = file_parent_dir + '/cstrip-fcn32s-color/test.prototxt'
hha2_weights = file_parent_dir + \
    '/cstrip-fcn32s-hha2/HHA2snapshot/secondTrain_lowerLR_iter_2000.caffemodel'
hha2_proto = file_parent_dir + '/cstrip-fcn32s-hha2/test.prototxt'
solver = caffe.SGDSolver(file_location + '/solver.prototxt')

# surgeries
color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
surgery.transplant(solver.net, color_net, suffix='color')
del color_net

hha2_net = caffe.Net(hha2_proto, hha2_weights, caffe.TEST)
surgery.transplant(solver.net, hha2_net, suffix='hha2')
del hha2_net

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt(
    file_location[:file_location.rfind('/')] + '/data/cs-trip/val.txt',
    dtype=str)
score.seg_tests(solver, False, val, layer='score')

# for _ in range(50):
#     print '------------------------------'
#     print 'Running solver.step iter {}'.format(_)
#     print '------------------------------'
#     solver.step(2000)
# score.seg_tests(solver, False, val, layer='score')
