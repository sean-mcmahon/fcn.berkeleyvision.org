# import caffe
import numpy as np
import os, sys
import imp
# add '../' directory to path for importing score.py and surgery.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/pkg/suse11/caffe/glog/0.3.3/lib/')

# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

# import support functions
caffe_root = '/home/n8307628/Fully-Conv-Network/Resources/caffe'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_cpu()
# with caffe found load remaining files (they use caffe too)
import surgery, score

weights = 'nyud-fcn32s-color-heavy.caffemodel'

# init
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
