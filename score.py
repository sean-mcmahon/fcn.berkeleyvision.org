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
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    cstrip_dir = home_dir + '/Construction_Site/Springfield/12Aug16/K2'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    cstrip_dir = home_dir + '/hpc-home/Construction_Site/Springfield/12Aug16/K2'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)


def fast_hist(a, b, n):
    # a is GT pixel values
    # b is binary class image of max predictions
    # n is number of classifications (=2)
    # This (below) removes dud labels? Yep, remove all labels less than 0 or
    # geater than number of classifications
    k = (a >= 0) & (a < n)
    # print 'a has {} values\nb has {} values\nand n is {}'.format(np.unique(a), np.unique(b),n)
    # print 'n * a[k].astype(int) + b[k] has values {}. a[k] has {}, b[k] has {}'.format(np.unique(n * a[k].astype(int) + b[k]),
    #                             np.unique(a[k]),np.unique(b[k]))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(net, save_dir, dataset, layer='score', gt='label', dataL='data'):
    n_cl = net.blobs[layer].channels  # channels is shape(1) of blob dim
    # n_cl number of classification channels? (2 for tripnet)
    if save_dir and not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        print '>> Foward pass for {} complete'.format(idx)
        # print '>> shape of score layer should be (2, 540, 960) and is:
        # {}'.format(np.shape(net.blobs[layer].data[0]))
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                          net.blobs[layer].data[0].argmax(0).flatten(),
                          n_cl)
        # print 'Hist format should be \n(num 0"s, num 1"s\nnum 2"s, num
        # 3"s)\nHist value is actually: \n{}\n'.format(hist)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(
                0).astype(np.uint8) * 255, mode='P')
            # im.save(os.path.join(save_dir, ''.join(idx) + '.png'))
            im_gt = Image.fromarray(
                net.blobs[gt].data[0, 0].astype(np.uint8) * 255, mode='P')
            # im_gt.save(os.path.join(save_dir, ''.join(idx) + '_GT.png'))
            try:
                colorArray = net.blobs[dataL].data[0].astype(np.uint8)
                colorArray = colorArray.transpose((1, 2, 0))
                colorArray = colorArray[..., ::-1]
                colorIm = Image.fromarray(colorArray)
            except:
                print '> Reading colour image from network failed, reading from file'
                colorIm = Image.open(
                    glob.glob('{}/{}/colour/colourimg_{}_*'.format(
                        cstrip_dir, idx[0], idx[1]))[0])
            overlay = Image.blend(colorIm.convert(
                "RGBA"), im.convert("RGBA"), 0.7)
            overlay.save(os.path.join(save_dir, ''.join(idx) + '.png'))
            gt_overlay = Image.blend(colorIm.convert(
                "RGBA"), im_gt.convert("RGBA"), 0.7)
            gt_overlay.save(os.path.join(save_dir, ''.join(idx) + '_GT.png'))
        # compute the loss as well
        try:
            loss += net.blobs['loss'].data.flat[0]
        except:
            print '> compute_hist: error calculating loss, probably no loss layer'
            loss += 0
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
    do_seg_tests(solver.test_nets[0], solver.iter,
                 save_format, dataset, layer, gt)


def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label', dataL='data'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    # TODO get better perfomance metrics
    # as I only care about trip detection performance
    print '> Computing Histagram'
    hist, loss = compute_hist(net, save_format, dataset, layer, gt, dataL)
    print '>>> Hist = {}'.format(hist)
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

    # trip hazard IU (label 1)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'trip IU', iu[1], 'non-trip IU', iu[0]
    print '>>>', datetime.now(), 'Iteration', iter, 'trip accuracy', acc[1], 'non-trip accuracy', acc[0]

    return hist
