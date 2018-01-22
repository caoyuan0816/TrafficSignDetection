"""
Traffic Signs Detection
CS 216 Final Project
Yuan Cao
X0971392
train.py
    - Classifier training methods
"""

import random
import time

from sklearn import svm
import skimage as ski
from skimage.feature import hog
import skimage.io as io

from detect import detect
from tools import isIntersect


def trainSVM(trainData, trainTarget, probability=True, kernel='rbf'):
    """
    Train classifer using SVM
    Parameters:
        - trainData: train data
        - trainTarget: target data
        - probability: whether or not open probability estimate, default=True
        - kernel: kernel type (candidates: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), default='rbf'
    Return:
        - a trained classifer
    """

    clf = svm.SVC(kernel=kernel, probability=probability)
    clf.fit(trainData, trainTarget)

    return clf


def testSVMbyPosData(clf, posFeature):
    """
    Test SVM Machine by using positive feature vectors.
    Parameters:
        - clf: classifer which want to be tested
        - posFeature: positive feature vectors
    Return:
        - result: a [0, 1] float value shows correct ratio of test
    """
    c = 0
    for pf in posFeature:
        if clf.predict([pf])[0] == -1:
            c = c + 1

    return c / len(posFeature)


def minePositiveNegtiveData(clf, negImages, labels, trainData, trainTarget, miningTime, w, h, n, scale, step):
    """
    Mine positive negtive data
    Parameters:
        - clf: current classifier
        - negImages: list of negative images
        - labels: labels of these negative images
        - trainData: old train data
        - trainTarget: old train target
        - miningTime: mining time
        - w: width of template
        - h: height of template
        - n: pyramid level number
        - scale: pyramid scale
        - step: sliding step
    Return:
        - nTrainData: new train data
        - nTrainTarget: new train target
    """
    print("...Mining Positive Negative Data:")
    t = 0
    for i in range(miningTime):
        # Randomly choose a negtive image
        iNegImage = random.randint(0, len(negImages)-1)
        ngI = negImages[iNegImage]

        # Detect it using current classifer
        detectionResult = detect(clf, ngI, w, h, n, scale, step)

        # Get all wrong answers
        for result in detectionResult:
            correctResult = list(filter(lambda x:x[0] == iNegImage , labels))
            intersection = False
            for c in correctResult:
                if isIntersect([c[1], c[2], c[3], c[4]], result):
                    intersection = True
                    break
            # Wrong answer
            if not intersection:
                t = t + 1
                # get feature vector of this wrong answer
                p = ngI[round(result[1]):round(result[1]+result[3]), round(result[0]):round(result[0]+result[2])]
                io.imsave('C:\\Yuan\\CS216\\FinalProject\\images\\Wrong\\{}.png'.format(time.time()), p)
                # scale patch to tamplate size
                ski.transform.resize(p, [h, w])
                feature = hog(p, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
                # add new negative data into old list
                trainData = trainData + [feature]
                trainTarget = trainTarget + [1]

    print("...{}, {}".format(len(trainData), len(trainTarget)))
    print("...Got {}".format(t))
    return trainData, trainTarget
