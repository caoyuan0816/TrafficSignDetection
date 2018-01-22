"""
Traffic Signs Detection
CS 216 Final Project
Yuan Cao
X0971392
tools.py:
    Useful tools for project.
"""

import math
import os

import numpy as np
import matplotlib.pyplot as pl
from scipy.ndimage.filters import correlate
import skimage.io as io


def isIntersect(r1, r2, returnArea=False):
    """
    Determaine 2 rectangles whether or not have intersection
    Parameters:
        - r1: rectangle 1
        - r2: rectangle 2
        - returnArea: whether or not return area of intersection
    Return:
        - True or False
    """
    left = max(r1[0], r2[0])
    right = min(r1[0]+r1[2], r2[0]+r2[2])
    top = max(r1[1], r2[1])
    bottom = min(r1[1]+r1[3], r2[1]+r2[3])

    if not returnArea:
        if left < right and top < bottom:
            return True
        return False
    else:
        if left < right and top < bottom:
            area = (right - left) * (bottom - top)
            return True, area
        return False, -1


def washTxtGt(FILE_PATH, curDataSet, w, h, sizeWash=True):
    """
    Transform a txt gt file to a list
    Parameters:
        - FILE_PATH: file path
    Return:
        - labels: a list saving content of file.
                  format: [imgNo, leftCol, topRow, rightCol, BottomRow, ClassID]
    """
    labels = []
    with open(FILE_PATH, 'r') as f:
        for line in f.readlines():
            line = line[:-1]
            l = line.split(';')
            l[0] = int(l[0][:-4])
            l[1] = int(l[1])
            l[2] = int(l[2])
            l[3] = int(l[3])-int(l[1])
            l[4] = int(l[4])-int(l[2])
            l[5] = int(l[5])
            if l[5] in curDataSet:
                if not sizeWash:
                    labels.append(l)
                if sizeWash and l[3] >= w and l[4] >= h:
                    labels.append(l)

    return labels


def averageBoxes(trainImages, scale, binSize=8):
    """
    Estimate a good template size for HOG template
    1. the aspect ratio should as close as possible to the original ratio
    2. the resolution should be close to the original one
    3. w, h % 8 == 0
    Parameter:
        - trainImages: list of training images
        - binSize: the block size of HOG function
    Return:
        - w: width
        - h: height
    """
    # average ratio
    aratio = 0
    # average area
    area = 0

    for im in trainImages:
        (height, width) = im.shape
        height = height * scale
        width = width * scale
        aratio  = aratio + width / height
        area = area + width * height
    aratio = aratio / len(trainImages)
    area = area / len(trainImages)

    barea = math.floor(area / (binSize**2))

    W = math.sqrt(aratio * barea)
    H = W / aratio

    w = round(W)*binSize
    h = round(H)*binSize

    return w, h


def getGradient(I, signed=False):
    """
    Given an image I, calculate the magnitude and orientation of gradient.
    Parameter:
        - I: Gray scale image I
        - signed: determain return signed orientation or unsigned
    Return:
        - mag: the magnitute of gradient
        - ori: the orientation of gradient
    """

    # Define filters
    fx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    fy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Calculate correlation
    dx = correlate(I, fx)
    dy = correlate(I, fy)

    # Calculate magnitude and orientation
    mag = np.sqrt(dx*dx, dy*dy)
    ori = np.arctan2(dx, dy)

    # unsigned
    if not signed:
        pi = np.pi
        ori[ori < -pi/2] = ori[ori < -pi/2] + pi
        ori[ori > pi/2] = ori[ori > pi/2] - pi

    return mag, ori
