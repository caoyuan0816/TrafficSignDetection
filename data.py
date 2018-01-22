"""
Traffic Signs Detection
CS 216 Final Project
Yuan Cao
X0971392
data.py
    - Data manipulation methods
"""

import os
import csv
import random
import pickle
import time
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.feature import hog
import skimage.io as io

from tools import averageBoxes, washTxtGt, isIntersect
from train import trainSVM, minePositiveNegtiveData

def loadInData(DATA_PATH, imageSet, t=0):
    """
    Load data from DATA_PATH
    Parameter:
        - DATA_PATH: path of data
        - imageSet: a list containing the sub data set which you want to use
        - t:
                0: positive data
                1: negative data
    Return:
        - images: the list of images
    """
    images = []
    if t == 0:
        for i in imageSet:
            CUR_PATH = os.path.join(DATA_PATH, '{:02d}'.format(i))
            for f in os.listdir(CUR_PATH):
                images.append(ski.color.rgb2gray(plt.imread(os.path.join(CUR_PATH, f))))
    if t == 1:
        for f in os.listdir(DATA_PATH):
            images.append(ski.color.rgb2gray(plt.imread(os.path.join(DATA_PATH, f))))

    return images


def calPositive(images, w, h, orientations, ppc, cpb):
    """
    Calculate positive training data using training imges as positive example
    Setting:
        - size of hog block: (8, 8)
        - orientations: 9
    Parameters:
        - images: a list of positive images (gray scale, double)
        - w: width of template
        - h: height of template
        - orientations: prientations
        - ppc: pixels per cell
        - cpb: cells per block
    Return:
        - posData: a list saving all positive feature vectors
    """

    # Rescale images
    for i in range(len(images)):
        images[i] = ski.transform.resize(images[i], [w, h])

    posData = []
    # Calculate feature vector of each training image
    for im in images:
        feature = hog(im, orientations=orientations, pixels_per_cell=(ppc, ppc),\
                                                     cells_per_block=(cpb, cpb))
        posData.append(feature)
    return posData


def calNegative(images, labels, n, w, h, oRate, orientations, ppc, cpb):
    """
    Calculate negative training data by random choose area is not ground truth
    Setting:
        - size of hog block: (8, 8)
        - orientations: 9
    Parameters:
        - images: a list of negative images
        - n: number of total negative data
        - w: the width of data
        - h: the height of data
    Return:
        - negData: a lost saving all negative feature vectors
    """

    # Randomly choose a image from images
    # and randomly choose a patch from that image
    # repeat n times

    def __patchGenerater():
        '''
        A generater generate a random patch each time.
        '''
        for _ in range(n):
            # Random image
            imageN = random.randint(0, len(images)-1)
            # Load in image
            im = images[imageN]
            (ih, iw) = im.shape
            # Get ground truth data of this image
            l = list(filter(lambda x: x[0] == imageN, labels))
            # Generate patch
            while True:
                xmin = random.randint(0, ih-h-1)
                ymin = random.randint(0, iw-w-1)
                intersection = False
                # Determaine intersections
                for r2 in l:
                    f, area = isIntersect([ymin, xmin, w, h], r2[1:], returnArea=True)
                    if f and area/w*h < oRate:
                        intersection = True
                        break
                if not intersection:
                    break
            patch = im[xmin:xmin+w, ymin:ymin+h]
            yield patch

    negData = []
    for p in __patchGenerater():
        feature = hog(p, orientations=orientations,\
                      pixels_per_cell=(ppc, ppc),
                      cells_per_block=(cpb, cpb))
        negData.append(feature)

    return negData
