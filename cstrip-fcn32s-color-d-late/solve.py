#! /usr/bin/python
"""
cstrip Color and HHA

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import surgery

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='CPU')
args = parser.parse_args()
print 'This is the colour-HHA solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
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
    print '-- GPU Mode Chosen -- {}'.format(args.mode)
    print '==============='


# You may want to change these initialisation weights,
# they differ slightly to the models being loaded
color_weights = file_parent_dir + \
    '/pretrained_weights/nyud-fcn32s-color-heavy.caffemodel'
color_proto = file_parent_dir + '/nyud-fcn32s-color/trainval.prototxt'
hha_weights = file_parent_dir + \
    '/pretrained_weights/nyud-fcn32s-hha-heavy.caffemodel'
hha_proto = file_parent_dir + '/nyud-fcn32s-hha/trainval.prototxt'
solver = caffe.SGDSolver(file_location + '/solver.prototxt')

# surgeries
color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
surgery.transplant(solver.net, color_net, suffix='color')
del color_net

hha_net = caffe.Net(hha_proto, hha_weights, caffe.TEST)
surgery.transplant(solver.net, hha_net, suffix='hha')
del hha_net

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)

# scoring
# test = np.loadtxt('../data/nyud/test.txt', dtype=str)

for _ in range(50):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(2000)
    # score.seg_tests(solver, False, val, layer='score')
