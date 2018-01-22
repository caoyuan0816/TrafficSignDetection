"""
Traffic Signs Detection
CS 216 Final Project
Yuan Cao
X0971392
detect.py
    - Detection methods
"""

import pickle
import os
import time
import warnings
from math import floor

import numpy as np
import skimage as ski
from skimage.feature import hog
from scipy.ndimage.filters import correlate
from sklearn import svm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import skimage.io as io
import tqdm

from tools import isIntersect, washTxtGt


def generateTestWindow(I, w, h, step, orientations, ppc, cpb):
    """
    A generator wich generate test windows.
    Method: pyramid and sliding window
    Parameters:
        - I: image I
        - w: width of window size
        - h: height of window size
        - step: step size of sliding
    Return:
        - the HOG feature vector of one of testing window
    """

    (Ih, Iw) = I.shape
    # Rescale the image to Iw % w == 0, Ih % h == 0
    Ih = round(Ih/h)*h
    Iw = round(Iw/w)*w

    I = ski.transform.resize(I, [Ih, Iw])
    featureI = hog(I, orientations=orientations,\
                   pixels_per_cell=(ppc, ppc),\
                   cells_per_block=(cpb, cpb))

    hogI = featureI.reshape((floor(Ih/ppc), floor(Iw/ppc), orientations))

    for x in range(0, Ih-h, step):
        for y in range(0, Iw-w, step):
            top = int(round(x/ppc))
            bot = int(top + h/ppc)
            left = int(round(y/ppc))
            right = int(left + w/ppc)

            feature = hogI[top:bot, left:right, :].flatten()
            yield x, y, feature


def generatePyramid(I, scale, n):
    """
    Generate pyramid of image I
    Parameters:
        - I: image I
        - scale: scale rate
        - n: number of level of pyramid
    Return:
        - I: new image I, after scaled
        - r: current scale rate
    """

    r = 1.0
    yield I, r
    t = 0
    for i in range(n-1):
        I = ski.transform.rescale(I, scale)
        r = r * scale
        yield I, r


def detect(clf, I, w, h, n, scale, step, orientations, ppc, cpb):
    """
    Detect objects using a trained svm file.
    Parameters:
        - SVM_PATH: svm file path
        - I: single test image
        - w: width of template
        - h: height of template
        - n: level number of pyramid
        - scale: scale rate of each level of pyramid
        - step: sliding window step size
    Return:
        - detectionResults: a list containing detection rectangle results
            format: [y, x, h, w, score]
    """

    detectionResults = []
    # pyramid
    for cI, r in generatePyramid(I, scale, n):
        #print('image size: {}'.format(cI.shape))
        # sliding windows
        for x, y, win in generateTestWindow(cI, w, h, step, orientations, ppc, cpb):
            # Positive detection
            if clf.predict([win])[0] == -1:
                result = list(map(lambda x: x/r, [y, x, h, w]))
                score = clf.predict_proba([win])[0][0]
                result.append(score)
                detectionResults.append(result)

    return detectionResults


def washDetectionResult(detectionResults, threshold, overlap):
    """
    Wash detection result using the probability value
    (Non-Maximum Suppression Problem)
    Parameters:
        - detectionResults: detection results
        - threshold: hard threshold for all results
        - overlap: overlap for NMS algorithm
    Return:
        - ndetectionResults: new detection results (after washing)
    """

    # Threshold
    detectionResults = [x for x in detectionResults if x[4] >= threshold]

    # Brute Force way
    '''
    resultGroup = []
    # Seperate results by group, inside a group, these rectangle have intersections
    for result in detectionResults:
        # Find intersections with rectangles inside result gruop
        flag = False
        for group in resultGroup:
            if flag:
                break
            for cr in group:
                if isIntersect(result, cr):
                    flag = True
                    group.append(result)
                    break
        if not flag:
            resultGroup.append([result])

    ndetectionResults = []
    # Washing result
    for group in resultGroup:
        # Washing by maximum probability
        #pmax = max([x[4] for x in group])
        #ndetectionResults.append(list(filter(lambda x:x[4]>=pmax, group))[0])
        # Washing by maximum size
        break
    '''

    # NMS-1 (Slow)

    # Sort result list by probability
    # Descend order
    detectionResults = sorted(detectionResults, key=lambda x:x[4], reverse=True)
    flags = [True for x in range(len(detectionResults))]

    for i in range(len(detectionResults)):
        # Big brother
        if flags[i]:
            # Get his little brothers
            for j in range(i+1, len(detectionResults)):
                inter, area = isIntersect(detectionResults[i], detectionResults[j], returnArea=True)
                if inter and area/min(detectionResults[i][2]*detectionResults[i][3],\
                                      detectionResults[j][2]*detectionResults[j][3]) >= overlap:
                    # Littel brother gone
                    flags[j] = False

    ndetectionResults = []
    # Save all Big brother
    for i in range(len(flags)):
        if flags[i]:
            ndetectionResults.append(detectionResults[i])

    return ndetectionResults


def showDetectionResult(I, detectionResults, w, h, showCorrectResult=False, labels=None, curFilename=None):
    """
    Show Detection results
    Parameters:
        - I: testing image
        - detectionResults: detection results
        - w: width of template
        - h: height of template
        - showCorrectResult: whether or not show correct answer
        - labels: if want to show correct answers, need provid labels
        - curFilename: if want to show correct answers, need provid cur image name
    """
    fig, ax = plt.subplots(1)

    ax.imshow(I)

    for rect in detectionResults:
        pr = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3],\
                                linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(pr)
        ax.text(rect[0], rect[1], "{:.3f}".format(rect[4]), color="blue", fontsize="10")

    if showCorrectResult:
        ll = list(filter(lambda x:x[0]==int(curFilename[:-4]), labels))
        for l in ll:
            pr = patches.Rectangle((l[1], l[2]), l[3], l[4],\
                                    linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(pr)

    ax.axis('off')
    plt.show()
    return


def doTest(FILE_PATH, fileName, clf, w, h, n, scale, step, threshold, labels,\
           raw_labels, overlap, WRONG_PATH, orientations, ppc, cpb):
    """
    Run detect method to one single testing images
    - Parameters:
        - TEST_PATH
    - Return:
        - tot: total number of detection is this image
        - suc: total successed detection number
    """
    tot = 0
    suc = 0

    I = ski.color.rgb2gray(plt.imread(FILE_PATH))
    detectionResults = detect(clf, I, w, h, n, scale, step, orientations, ppc, cpb)
    detectionResults = washDetectionResult(detectionResults, threshold, overlap)

    ll = list(filter(lambda x:x[0] == int(fileName[:-4]), labels))
    rl = list(filter(lambda x:x[0] == int(fileName[:-4]), raw_labels))

    for r in detectionResults:
        flag = False
        for l in ll:
            inter, area = isIntersect(l[1:], r, returnArea=True)
            if area/min(l[3]*l[4], r[2]*r[3]) >= 0.6:
                    flag = True
                    suc = suc + 1
                    break
        if not flag:
            ff = False
            for l in rl:
                inter, area = isIntersect(l[1:], r, returnArea=True)
                if area/min(l[3]*l[4], r[2]*r[3]) >= 0.6:
                    ff = True
                    break
            if ff:
                continue
            p = I[round(r[1]):round(r[1]+r[3]), round(r[0]):round(r[0]+r[2])]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                io.imsave(os.path.join(WRONG_PATH, '{}.png'.format(time.time())), p)
        tot = tot + 1
    # -------------- whether or not show visualization test result
    #showDetectionResult(plt.imread(FILE_PATH), detectionResults, w, h, True, labels, fileName)

    return tot, suc
