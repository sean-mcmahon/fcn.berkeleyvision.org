import numpy as np
from PIL import Image
import scipy.io
import glob
import os
from os.path import expanduser, join, dirname, realpath
from os import getcwd
import imp

home_dir = expanduser("~")
if 'n8307628' in home_dir:
    caffe_root = home_dir+'/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir+'/src/caffe'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
import random

class CStripSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from Construction Site Trip
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels consist of trip (1) and non-trip (2)

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - cstrip_dir: path to CS_trip dir
        - split: train / val / test
        - tops: list of tops to output from {color, depth, hha, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for trip hazard semantic segmentation.

        example: params = dict(cstrip_dir="/path/to/cstrip_dir", split="val",
                               tops=['color', 'hha', 'label'])
        """
        # config
        print 'cs_trip_layer: beginning setup'
        params = eval(self.param_str)
        if 'sean' in expanduser("~"):
            self.cstrip_dir = '/home/sean/hpc-home'+params['cstrip_dir']
        else:
            self.cstrip_dir = '/home/n8307628'+params['cstrip_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        # self.null_value = params.get('null_value',-1)
        self.file_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # store top data for reshape + forward
        self.data = {}

        # Null value for when depth pixels are 0
        # - log(0) gives -inf which makes the CNN ouput NaNs
        self.null_value = -1
        print '***********************'
        print 'null_value set to {}'.format(self.null_value)
        print '***********************'

        # TODO: Find means of images in CS dataset
        self.mean_bgr = np.array((0,0,0), dtype=np.float32)
        self.mean_hha = np.array((0,0,0), dtype=np.float32)
        self.mean_logd = np.array((0,), dtype=np.float32)

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '{}/{}.txt'.format(self.file_location+'/data/cs-trip', self.split)
        dir_indices_list = open(split_f, 'r').read().splitlines()
        # Because my txt file has layout 'sub_dir idx\n' I have to do some more
        # parsing, not the prettiest way but it'll work
        self.indices = []; self.sub_dir = []
        for item in dir_indices_list:
            split_list = item.split(' ')
            self.sub_dir.append(split_list[0])
            self.indices.append(split_list[1])
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)
        print 'cs_trip_layers: setup complete.'

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.indices[self.idx], self.sub_dir[self.idx])
            top[i].reshape(1, *self.data[t].shape)
        # print 'cs_trip_layers: Reshaped top {}'.format(top[:])

    def forward(self, bottom, top):
        # print 'cs_trip_layers: forward method: top {}'.format(top[:])
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        # print 'cs_trip_layers: forward complete.'

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx, sub_dir):
        # print 'cs_trip_layers: Loading top {}'.format(top)
        if top == 'color':
            return self.load_image(idx, sub_dir)
        elif top == 'label':
            return self.load_label(idx, sub_dir)
        elif top == 'depth':
            return self.load_depth(idx, sub_dir)
        elif top == 'hha':
            return self.load_hha(idx, sub_dir)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_image(self, idx, sub_dir):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        # idx_str = str(idx).zfill(4)
        im = Image.open(glob.glob('{}/{}/colour/colourimg_{}_*'.format(self.cstrip_dir, sub_dir, idx))[0])
        # im = Image.open('{}/{}/colour/colourimg{}.png'.format(self.cstrip_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))
        # print 'cs_trip_layers: colour image loaded shape={}'.format(np.shape(in_))
        if self.split is not 'train':
            print 'loading image from {} with index {}'.format(sub_dir,idx)
        return in_

    def load_label(self, idx, sub_dir):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        # generated these segmentation .mat files using export_depth_and_labels.m (hpc-cyphy/Datasets/NYU2)
        label = scipy.io.loadmat(glob.glob('{}/{}/labels/colourimg_{}_*'.format(self.cstrip_dir, sub_dir, idx))[0])['binary_labels'].astype(np.uint8)
        # label -= 1  # rotate labels
        label = label[np.newaxis, ...]
        # print 'cs_trip_layers: Label loaded, shape {}, has values {} and id {}/{}'.format(np.shape(label), np.unique(label),sub_dir, idx)
        return label

    def load_depth(self, idx, sub_dir):
        """
        Load pre-processed depth for my CS trip hazard segmentation set.
        """

        im = Image.open(glob.glob('{}/{}/depth/depthimg_{}_*'.format(self.cstrip_dir, sub_dir, idx))[0])
        d = np.array(im, dtype=np.float32)
        # print 'depth pixel values before log are: {}\nMin {}, max {} and shape {}'.format(np.unique(d), min(d.flatten()),max(d.flatten()), np.shape(d))
        d = np.log(d)
        d[np.isneginf(d)] = self.null_value
        d -= self.mean_logd
        d = d[np.newaxis, ...]
        # print 'depth pixel values are {}\nMin {}, max {} and shape {}'.format(np.unique(d), min(d.flatten()),max(d.flatten()), np.shape(d))
        return d

    def load_hha(self, idx, sub_dir):
        """
        Load HHA features from Gupta et al. ECCV14.
        See https://github.com/s-gupta/rcnn-depth/blob/master/rcnn/saveHHA.m
        """
        im = Image.open('{}/data/hha/img_{}.png'.format(self.cstrip_dir, idx))
        hha = np.array(im, dtype=np.float32)
        hha -= self.mean_hha
        hha = hha.transpose((2,0,1))
        return hha
