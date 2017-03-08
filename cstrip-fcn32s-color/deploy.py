
# This import might casuse trouble, my network requires a custom version
# of caffe
import sys
import numpy as np
from PIL import Image
import os
import glob
import imp
import time
import scipy
try:
    import cv2
except:
    print 'failed to load opencv2'
else:
    print 'opencv2 sucessfully loaded'

home_dir = os.path.expanduser("~")
if 'n8307628' in home_dir:
    caffe_root = os.path.join(home_dir, 'Fully-Conv-Network/Resources/caffe')
    base_dir = '/home/n8307628/'
    sys.path.append(os.path.join(base_dir, 'Dropbox/Uni/Code/FCN_models'))
elif 'sean' in home_dir:
    caffe_root = os.path.join(home_dir, 'src/caffe')
    base_dir = '/home/sean/hpc-home/'
    sys.path.append(os.path.join(base_dir,
                                 'Fully-Conv-Network/Resources/FCN_models'))
else:
    print 'unknown directory'
    raise
from score import fast_hist, compute_flagMetric
print 'loading caffe from ', caffe_root
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)


def prepImage(img):
    # perform image preprocessing for CNN with caffe

    # no mean subtraction for now
    mean_bgr = np.array((0, 0, 0), dtype=np.float32)
    n_img = np.array(img, dtype=np.float32)
    # make bgr, may not be necessary
    n_img = n_img[:, :, ::-1]
    # mean subtract (nothing subtracted for now)
    n_img -= mean_bgr
    # reshape from w*h*3 to 3*w*h (or 3*h*w??)
    n_img = n_img.transpose((2, 0, 1))

    return n_img


def load_label(label_name):
    return scipy.io.loadmat(label_name)['binary_labels'].astype(np.uint8)


def deploy(net, data, visualise=True, image_name='overlay_image'):
    score_layer = 'softmax_score'
    data_layer = 'data'
    in_img = prepImage(data)
    # reshape input for any sized image (data blob is N x C x H x W)
    net.blobs[data_layer].reshape(1, *in_img.shape)
    net.blobs[data_layer].data[...] = in_img
    start_time = time.time()
    net.forward()
    print 'foward pass took {} seconds'.format(time.time() - start_time)

    if visualise:
        out = net.blobs[score_layer].data[0].argmax(axis=0)
        # out = net.blobs[score_layer].data[0][1]
        sums = np.sum(net.blobs[score_layer].data[0], axis=0)
        # print 'out[0] shape ', np.shape(net.blobs[score_layer].data[0][1])
        # print 'out[0][0] unique', np.unique(out)
        # print 'softmax sum (should be all 1 ) shape', np.shape(
        #     sums), 'unique=', np.unique(sums)
        # give array img values, do i need to convert to BGR?
        out_img = out.astype(np.uint8) * 255

        # load img as PIL
        img = np.array(data, dtype=np.uint8)
        # print 'img[0][0] unique', np.unique(img)
        colorIm = Image.fromarray(img)
        # Network prediction as PIL image
        im = Image.fromarray(out_img, mode='P')
        overlay = Image.blend(colorIm.convert(
            "RGBA"), im.convert("RGBA"), 0.7)
        basename = os.path.splitext(os.path.basename(image_name))[0]

        if 'n8307628' in home_dir:
            overlay.save(os.path.join(home_dir, 'baysian_cnn_images',
                                      basename + '_overlay.png'))
        else:
            cv_overlay = np.array(overlay)
            cv2.imshow('Image', out_img)
            cv2.waitKey(0)
            cv2.imshow('Image2', cv_overlay)
            cv2.waitKey(0)
        # convert from rgb to bgr - needed?
        # cv_overlay = cv_overlay[:, :, ::-1].copy()
    return net

if __name__ == '__main__':
    file_location = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # load data and label names
    test_dir = os.path.join(
        base_dir, 'Construction_Site/Springfield/12Aug16/K2/2016-08-12-10-09-26_groundfloorCarPark/')
    img_names = glob.glob(os.path.join(
        test_dir, 'labelled_colour', '*.png'))
    img_names.sort()
    label_names = glob.glob(os.path.join(test_dir, 'labels', '*.mat'))
    label_names.sort()
    # initialise network
    arch = os.path.join(file_location, 'deploy_col.prototxt')
    weights = os.path.join(
        base_dir, 'Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color/colorSnapshot/_iter_6000.caffemodel')

    # Initialise networks
    caffe.set_mode_cpu()
    test_net = caffe.Net(arch, weights, caffe.TEST)
    train_net = caffe.Net(arch, weights, caffe.TRAIN)

    # allocate memory for loop
    num_images = len(img_names)
    train_hist = np.zeros((2, 2))
    test_hist = np.zeros((2, 2))
    for count, (img_name, label_name) in enumerate(zip(img_names, label_names)):
        image = Image.open(img_name)
        label = load_label(label_name).flatten()
        # print 'loaded image ', os.path.basename(img_name), ' has shape ', np.shape(image)
        # forward pass
        print '\n--- foward pass', count + 1, 'of', num_images, '---'
        num_loops = 2
        score_blobs = np.zeros((960, 540, 2, num_loops))
        basename = os.path.splitext(os.path.basename(img_name))[0]
        for i in range(num_loops):
            out_name = basename + '_' + str(i) + 'th_repeat'
            train_net = deploy(
                train_net, image, visualise=False, image_name=out_name)
            score_blobs[:, :, :, i] = train_net.blobs['softmax_score'].data[0]
        mean_scores = np.mean(score_blobs, axis=3)
        train_hist += fast_hist(label,
                                mean_scores.argmax(0).flatten(), 2)

        test_net = deploy(test_net, image, visualise=True, image_name=img_name)
        test_hist += fast_hist(label,
                               test_net.blobs['softmax_score'].data[0], 2)
        break
    test_acc = np.diag(test_hist).sum() / test_hist.sum()
    print '>>> Test acc ', test_acc
    train_acc = np.diag(train_hist).sum() / train_hist.sum()
    print '>>> Train acc'

    test_precision = test_hist[1, 1] / test_hist.sum(0)[1]
    test_recall = test_hist[1, 1] / test_hist.sum(1)[1]
    test_Fone = ((test_recall * test_precision) / (test_recall + test_precision)) * 2
    print '>>> test_recall (swapped)', test_recall
    print '>>> test_precision ', test_precision
    print '>>> test F1-score', test_Fone

    train_precision = train_hist[1, 1] / train_hist.sum(0)[1]
    train_recall = train_hist[1, 1] / train_hist.sum(1)[1]
    train_Fone = ((train_recall * train_precision) / (train_recall + train_precision)) * 2
    print '>>> train_recall (swapped)', train_recall
    print '>>> train_precision ', train_precision
    print '>>> test F1-score', train_Fone
