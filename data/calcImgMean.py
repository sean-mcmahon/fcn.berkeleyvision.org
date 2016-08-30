#! /usr/bin/python
"""
@author: Sean McMahon
"""
import numpy as np
import argparse
import glob
import os.path
from PIL import Image
import sys
from os.path import expanduser
import os


file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location)
home_dir = expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='/home/sean/hpc-home/Construction_Site/Springfield/12Aug16/K2')
parser.add_argument('--split', default='train')
parser.add_argument('--textfile_dir', default='/cs-trip')
args = parser.parse_args()
data_dir = args.data_dir
data_split = args.split
txt_dir = args.textfile_dir
dataset_indecies = np.loadtxt(file_location + txt_dir +
                              '/' + data_split + '.txt',
                              dtype=str)
num_colour_pixels = 0
colour_pixel_sum = 0
depth_pixel_sum = 0
num_depth_pixels = 0
hha_pixel_sum = 0
num_hha_pixels = 0
for idx in dataset_indecies:
    colourimg = Image.open(glob.glob(
        '{}/{}/colour/colourimg_{}_*'.format(data_dir, idx[0], idx[1]))[0])
    npColourImg = np.array(colourimg, dtype=np.float32)
    npColourImg = npColourImg[:, :, ::-1]  # network takes bgr values
    colour_pixel_sum += np.sum(np.sum(npColourImg, axis=0), axis=0)
    num_colour_pixels += npColourImg.sum()

    depthimg = Image.open(glob.glob(
        '{}/{}/depth/depthimg_{}_*'.format(data_dir, idx[0], idx[1]))[0])
    npDepthImg = np.array(depthimg, dtype=np.float32)
    npDepthImg = np.log(npDepthImg)
    npDepthImg[np.isneginf(npDepthImg)] = 0
    depth_pixel_sum += np.sum(np.sum(npDepthImg, axis=0), axis=0)
    num_depth_pixels += npDepthImg.sum()

    HHAimg = Image.open(glob.glob(
        '{}/{}/HHA/HHAimg_{}_*'.format(data_dir, idx[0], idx[1]))[0])
    npHHAImg = np.array(HHAimg, dtype=np.float32)
    hha_pixel_sum += np.sum(np.sum(npHHAImg, axis=0), axis=0)
    num_hha_pixels += npHHAImg.sum()

colour_mean = colour_pixel_sum / num_colour_pixels
depth_mean = depth_pixel_sum / num_depth_pixels
hha_mean = hha_pixel_sum / num_hha_pixels
print 'colour mean: ', colour_mean
print 'log(depth) mean: ', depth_mean
print 'hha mean: ', hha_mean
