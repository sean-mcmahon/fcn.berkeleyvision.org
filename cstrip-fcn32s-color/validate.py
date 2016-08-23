#! /usr/bin/python
"""
cstrip validate COLOUR only

"""
# import caffe
import numpy as np
import os, sys
from os.path import expanduser
import imp
import argparse
import glob

def add_slash(mystring):
    if mystring.endswith('/'):
        return mystring
    else:
        return mystring+'/'

# add '../' directory to path for importing score.py, surgery.py and pycaffe layer
file_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--iteration',default=8000)
parser.add_argument('--test_type', default='val')
parser.add_argument('--network_dir', default='cstrip-fcn32s-color')
args = parser.parse_args()
iteration = args.iteration
network_dir = args.network_dir
network_dir = add_slash(network_dir) # ensure slash present
print 'This is the COLOUR only validation!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir+'/Fully-Conv-Network/Resources/caffe'
    weight_dir = home_dir+ '/Fully-Conv-Network/Resources/FCN_models/' + network_dir
    snapshot_dir = glob.glob(weight_dir+'*napshot*')
    weights = snapshot_dir[0]+'/_iter_'+ str(iteration) +'.caffemodel'
elif 'sean' in home_dir:
    caffe_root = home_dir+'/src/caffe'
    weight_dir = home_dir+'/hpc-home/Fully-Conv-Network/Resources/FCN_models/'+ network_dir
    snapshot_dir = glob.glob(weight_dir+'*napshot*')
    weights = snapshot_dir[0]+'/_iter_'+ str(iteration) +'.caffemodel'
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

# init
solver = caffe.SGDSolver(file_location+'/solver.prototxt')
print 'glob snapshot folderoutput {}'.format(snapshot_dir[0])
solver.net.copy_from(weights)


if args.test_type=='val':
    test_set = np.loadtxt(file_location[:file_location.rfind('/')]+'/data/cs-trip/val.txt', dtype=str)
elif args.test_type=='test':
    test_set = np.loadtxt(file_location[:file_location.rfind('/')]+'/data/cs-trip/test.txt', dtype=str)
elif args.test_type=='train':
    test_set = np.loadtxt(file_location[:file_location.rfind('/')]+'/data/cs-trip/train.txt', dtype=str)
else:
    print 'Incorrect test_type given {}; expecting "train", "val" or "test"'.format(args.test_type)
    raise

print '\n>>>> Validation <<<<\n'
score.seg_tests(solver, file_location+'/images', test_set, layer='score')
