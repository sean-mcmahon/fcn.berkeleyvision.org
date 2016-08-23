from __future__ import division
import imp
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
from os.path import expanduser
import glob

home_dir = expanduser("~")
if 'n8307628' in home_dir:
    caffe_root = home_dir+'/Fully-Conv-Network/Resources/caffe'
    cstrip_dir = home_dir+'/Construction_Site/Springfield/12Aug16/K2'
elif 'sean' in home_dir:
    caffe_root = home_dir+'/src/caffe'
    cstrip_dir = home_dir+'/hpc-home/Construction_Site/Springfield/12Aug16/K2'
filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
caffe = imp.load_module('caffe', filename, path, desc)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir and not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        print '>> Foward pass for {} complete'.format(idx)
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8)*255, mode='P')
            # im.save(os.path.join(save_dir, ''.join(idx) + '.png'))
            im_gt = Image.fromarray(net.blobs[gt].data[0, 0].astype(np.uint8)*255, mode='P')
            # im_gt.save(os.path.join(save_dir, ''.join(idx) + '_GT.png'))
            colorIm = Image.open(glob.glob('{}/{}/colour/colourimg_{}_*'.format(cstrip_dir, idx[0], idx[1]))[0])
            overlay = Image.blend(colorIm.convert("RGBA"),im.convert("RGBA"),0.7)
            overlay.save(os.path.join(save_dir, ''.join(idx) + '.png'))
            gt_overlay = Image.blend(colorIm.convert("RGBA"),im_gt.convert("RGBA"),0.7)
            gt_overlay.save(os.path.join(save_dir, ''.join(idx) + '_GT.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
        # score_values = net.blobs[layer].data[0].argmax(0)
        # gt_values = net.blobs[gt].data[0, 0]
        # print '> score_values has shape ', np.shape(score_values)
        # print '> Unqiue score_values values: ', np.unique(score_values)
        # print '> gt_values has shape ', np.shape(gt_values)
        # print '> Unqiue gt_values values: ', np.unique(gt_values)
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    print '> Computing Histagram'
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()

    return hist
