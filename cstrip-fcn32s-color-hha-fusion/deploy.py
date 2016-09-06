#! /usr/bin/python
"""
by Sean McMahon

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
import surgery
import score
import nets
import glob
from PIL import Image


# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer

file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")
# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
from caffe.proto import caffe_pb2

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='GPU')
parser.add_argument('--iteration', default=8000)
parser.add_argument('--data_split', default='val')
parser.add_argument('--snapshot_filter', default='train')
args = parser.parse_args()
iteration = args.iteration
snapshot_filter = args.snapshot_filter
data_split = args.data_split

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

# Create fusion_test prototxt files
[fixed_net_path, conv_net_path] = nets.createNets(data_split)
dataset_split = np.loadtxt(
    file_parent_dir + '/data/cs-trip/{}.txt'.format(data_split), dtype=str)

fixed_fusion_net = caffe.Net(fixed_net_path, caffe.TEST)
interp_layers = [k for k in fixed_fusion_net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(fixed_fusion_net, interp_layers)

weight_dir = file_location
snapshot_dir = glob.glob(weight_dir + '/*napshot*')
weights = snapshot_dir[0] + '/' + snapshot_filter + \
    '_iter_' + str(iteration) + '.caffemodel'
convFusionNet = caffe.Net(conv_net_path, weights,  caffe.TEST)

print '\n------------------------------'
print 'Testing fixed_fusion_net'
print '------------------------------'
score.do_seg_tests(fixed_fusion_net, 0, os.path.join(
    file_location, data_split + '_fixedNet_images'), dataset_split,
    layer='score', gt='label', dataL='in_data')

print '\n------------------------------'
print 'Testing convFusionNet'
print '------------------------------'
score.do_seg_tests(convFusionNet, iteration, os.path.join(
    file_location, data_split + '_convNet_images'), dataset_split,
    layer='score', gt='label', dataL='in_data')
