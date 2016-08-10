import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'nyud-fcn32s-color-heavy.caffemodel'

# init
# caffe_root = '/home/sean/src/caffe'
# filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
# caffe = imp.load_module('caffe', filename, path, desc)
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/home/sean/hpc-cyphy/Datasets/NYU2/val.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    score.seg_tests(solver, False, val, layer='score')
