from __future__ import division
import imp
import numpy as np
import os
from datetime import datetime
from PIL import Image
from os.path import expanduser
import glob
import matplotlib
matplotlib.use('Agg')
from scipy.io import savemat, loadmat
import time

home_dir = expanduser("~")
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    cstrip_dir = home_dir + '/Construction_Site/Springfield/12Aug16/K2'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    cstrip_dir = home_dir + '/hpc-home/Construction_Site/Springfield/12Aug16/K2'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)


def append_hist(prev_hist, gt_blob_data, score_blob_data, num_classes):
    # Might not be working!
    threshlold_interval = 1
    thresholds = []
    hist_list = []
    for count in range(threshlold_interval, 100, threshlold_interval):
        threshold = count / 100.0
        thresholds.append(np.copy(threshold))  # as percentage
        thres_scores = score_blob_data >= threshold
        thres_scores = thres_scores.astype(int).flatten()
        histagram = fast_hist(gt_blob_data, thres_scores, num_classes)
        hist_list.append(np.copy(histagram))
    if prev_hist:
        for count, item in enumerate(prev_hist):
            hist_list[count] += item
    return hist_list, thresholds


def compute_flagMetric(score, folder, index, gt=False):
    matfilename = glob.glob('{}/{}/labels/colourimg_{}_*'.format(
        cstrip_dir, folder, index))[0]
    matfile = loadmat(matfilename)
    # get binary image of each trip label, assumes each label is of a
    # different trip hazard
    rawmasks = matfile['objects'][0][0][2][0]
    score = score.flatten()
    scoretrips = np.where(score > 0)[0]
    flagTp = 0
    flagFn = 0
    for mask in rawmasks:
        mask = mask.flatten()
        masktrips = np.where(mask > 0)[0]
        intersect = np.intersect1d(scoretrips, masktrips)
        if intersect.size != 0:
            flagTp += 1
        else:
            flagFn += 1
    if gt is not False:
        bin_mask = matfile['binary_labels']
        if np.array_equal(gt, bin_mask):
            # print 'arrays are identical'
            pass
        else:
            print 'arrays are not equal'
    return np.array((flagTp, flagFn))


def fast_hist(a, b, n):
    # a is GT pixel values
    # b is binary class image of max predictions
    # n is number of classifications (=2)
    # This (below) removes dud labels? Yep, remove all labels less than 0 or
    # geater than number of classifications
    k = (a >= 0) & (a < n)
    # print 'a has {} values\nb has {} values\nand n is
    # {}'.format(np.unique(a), np.unique(b),n)
    # print 'n * a[k].astype(int) + b[k] has values {}. \
    #         a[k] has {}, b[k] has {}'.format(np.unique(
    #         n * a[k].astype(int) + b[k]),
    #                             np.unique(a[k]),np.unique(b[k]))
    # hist = [No. true Neg , No. false Pos;
    #         No. false Neg, No. true Pos]
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(net, save_dir, dataset, layer='score', gt='label',
                 dataL='data'):
    n_cl = net.blobs[layer].channels  # channels is shape(1) of blob dim
    # n_cl number of classification channels? (2 for tripnet)
    if save_dir and not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_mat = False
    hist = np.zeros((n_cl, n_cl))
    Fmetrics = np.array((0, 0))
    loss = 0
    # threshold_hists = []
    forward_times = np.zeros((len(dataset), 1))
    for count, idx in enumerate(dataset):
        start_time = time.time()
        net.forward()
        forward_times[count] = time.time() - start_time
        # print '>> Foward pass for {} complete'.format(idx)
        # print '>> shape of score layer should be (2, 540, 960) and is:
        # {}'.format(np.shape(net.blobs[layer].data[0]))
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                          net.blobs[layer].data[0].argmax(0).flatten(),
                          n_cl)
        Fmetrics += compute_flagMetric(net.blobs[layer].data[0].argmax(0), idx[0],
                                       idx[1], gt=net.blobs[gt].data[0, 0])
        # threshold_hists, thresholds = append_hist(threshold_hists,
        #                                           net.blobs[gt].data[
        #                                               0, 0].flatten(),
        #                                           net.blobs[layer].data[
        #                                               0][1].flatten(),
        #                                           n_cl)
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
            if save_mat:
                score_blob = net.blobs[layer].data[0]
                label_blob = net.blobs[gt].data[0]
                matfilename = os.path.join(save_dir, ''.join(idx) + '.mat')
                # print '>>>>>
                # np.unqiue(score_blob)={}'.format(np.unique(score_blob))
                save_dict = {'score_blob': score_blob,
                             'label_blob': label_blob}
                savemat(matfilename, save_dict)
        # compute the loss as well
        try:
            loss += net.blobs['loss'].data.flat[0]
        except:
            print '> compute_hist: error calculating loss, probably no loss layer'
            loss += 0

    # precision, recall = compute_PR(threshold_hists, thresholds, save_dir)
    mean_run_time = forward_times.sum() / len(forward_times)
    return hist, loss / len(dataset), Fmetrics, mean_run_time

def seg_loss_tests(solver, dataset, layer='score', gt='data',
              dataL='data', test_type='val'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    seg_loss(solver.test_nets[0], solver.iter,
                 dataset,test_type, True, gt, layer)

def seg_loss(net, iteration, dataset, test_type='training',
             calc_hist=False, gt='data', layer='score'):
    print '> Computing Loss'
    loss = 0
    # threshold_hists = []
    forward_times = np.zeros((len(dataset), 1))
    if calc_hist:
        n_cl = net.blobs[layer].channels
        hist = np.zeros((n_cl, n_cl))
    for count, _ in enumerate(dataset):
        start_time = time.time()
        net.forward()
        forward_times[count] = time.time() - start_time
        if calc_hist:
            hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                              net.blobs[layer].data[0].argmax(0).flatten(),
                              n_cl)
        try:
            loss += net.blobs['loss'].data.flat[0]
        except:
            print '> compute_hist: error calculating loss, probably no loss layer'
            loss += 0
    mean_time = forward_times.sum() / len(forward_times)
    # mean loss, this prinout is tweaked to match caffe prinout for string
    # parsing
    print '>>>', datetime.now(), 'Iteration', '{}'.format(iteration), \
        test_type, 'set loss =', loss
    print '>>>', datetime.now(), 'Iteration', '{}'.format(iteration), \
        test_type, 'set Mean runtime =', mean_time
    if calc_hist:
        acc = np.diag(hist) / hist.sum(1)
        print '>>>', datetime.now(), 'Iteration', '{}'.format(iteration), \
            test_type, 'trip accuracy', acc[1]


def seg_tests(solver, save_format, dataset, layer='score', gt='label',
              dataL='data'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter,
                 save_format, dataset, layer, gt, dataL)


def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label',
                 dataL='data'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    print '> Computing Histagram'
    # hist = [No. true Neg , No. false Pos;
    #         No. false Neg, No. true Pos]
    hist, loss, Flags, mean_run_time = compute_hist(
        net, save_format, dataset, layer, gt, dataL)
    print '>>> Hist = {}'.format(hist)
    # mean loss, this prinout is tweaked to match caffe prinout for string
    # parsing
    print '>>>', datetime.now(), 'Iteration', '{},'.format(iter), 'loss =', loss
    # mean forward pass times
    print '>>>', datetime.now(), 'Iteration', iter, 'mean forward', mean_run_time
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
    print '>>>', datetime.now(), 'Iteration', iter, 'trip accuracy', acc[1], \
        'non-trip accuracy', acc[0]

    precision = hist[1, 1] / hist.sum(0)[1]  # hist[1,1] / (hist[1,0] + hist[1,1])
    # hist[1,1] / (hist[0,1] + hist[1,1])
    recall = hist[1, 1] / hist.sum(1)[1]
    Fone = ((recall * precision) / (recall + precision)) * 2
    print '>>>', datetime.now(), 'Iteration', iter, 'recall (swapped)', recall
    print '>>>', datetime.now(), 'Iteration', iter, 'precision (swapped)', precision
    print '>>>', datetime.now(), 'Iteration', iter, 'F1-score', Fone

    Fmetric_acc = Flags[0].astype(np.float) / Flags.sum().astype(np.float)
    print '>>>', datetime.now(), 'Iteration', iter, 'Flag Metric Acc', Fmetric_acc, \
        '\nFlags= ', Flags

    return hist
