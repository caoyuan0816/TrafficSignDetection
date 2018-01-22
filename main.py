"""
Traffic Signs Detection
CS 216 Final Project
Yuan Cao
X0971392
main.py
    - Driver script of project
"""

import os
import pickle
import configparser
import time
import ast
import argparse
import multiprocessing

import tqdm
import skimage as ski
from skimage.feature import hog

from data import *
from tools import *
from train import *
from detect import *


# Set running path
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Agr parser setting
parser = argparse.ArgumentParser(description='A Traffic Sign Detection Program. Author: Yuan Cao <caoyuan0816@gmail.com>')
parser.add_argument('action', help="The action you want to do. [train | mine | test]")
parser.add_argument("-c", help="Set the path of configure file. Please provide absolutly path.")

def init(CONFIG_PATH):
    """
    Init enviroments of project.
    """

    print("\n... Using config file: {}\n".format(CONFIG_PATH))

    # Load in config
    conf = configparser.ConfigParser()
    conf.read(CONFIG_PATH)
    configs = {}

    # Path init
    IMAGE_PATH = os.path.join(ROOT_PATH, conf['data_path']['IMAGE_PATH'])
    RESULT_PATH = os.path.join(ROOT_PATH, conf['data_path']['RESULT_PATH'])

    configs['SVM_PATH'] = os.path.join(ROOT_PATH, conf['data_path']['SVM_PATH'])
    configs['POS_PATH'] = os.path.join(IMAGE_PATH, conf['data_path']['POS_DIR'])
    configs['NEG_PATH'] = os.path.join(IMAGE_PATH, conf['data_path']['NEG_DIR'])
    configs['TEST_PATH'] = os.path.join(IMAGE_PATH, conf['data_path']['TEST_DIR'])
    configs['GT_PATH'] = os.path.join(IMAGE_PATH, 'gt.txt')

    # Runing dataset
    dataSet = ast.literal_eval(conf['data_set']['DEFINATION'])
    configs['curDataSet'] = dataSet[conf['data_set']['RUNNING_DATASET']]

    # Train configures
    configs['templateScale'] = float(conf['train']['TEMPLATE_SCALE'])
    configs['negFeatureVectorNum'] = int(conf['train']['NEG_FEATURE_VECTOR_NUM'])

    # HOG settings
    configs['orientations'] = int(conf['hog']['orientations'])
    configs['ppc'] = int(conf['hog']['ppc'])
    configs['cpb'] = int(conf['hog']['cpb'])

    # SVM settings
    configs['saveFileName'] = conf['svm']['SAVING_FILE_NAME']
    configs['loadFileName'] = conf['svm']['LOAD_FILE_NAME']
    configs['kernel'] = conf['svm']['KERNEL']
    configs['probability'] = bool(conf['svm']['PROBABILITY'])

    # Test settings
    configs['pLevel'] = int(conf['test']['PYRAMID_LEVEL'])
    configs['pScale'] = float(conf['test']['PYRAMID_SCALE'])
    configs['wStep'] = int(conf['test']['SLIDING_WINDOW_STEP'])
    configs['pThreshold'] = float(conf['test']['DETECTION_PROBABILITY_THRESHOLD'])
    configs['oRate'] = float(conf['test']['OVERLAP_RATE'])
    configs['WRONG_PATH'] = os.path.join(IMAGE_PATH, conf['test']['WRONG_DETECTION_DIR'])

    # Reinforce
    configs['rTime'] = int(conf['reinforce']['REINFORCE_TIME'])

    return configs


def train(POS_PATH, NEG_PATH, GT_PATH, curDataSet, templateScale,\
         orientations, ppc, cpb, negFeatureVectorNum, SVM_PATH, saveFileName,\
         kernel, probability, oRate):
    """
    Training SVM Classifier.
    """
    # load in training data and labels
    print("... Loading in training data")
    posImages = loadInData(POS_PATH, curDataSet, 0)
    negImages = loadInData(NEG_PATH, None, 1)
    # get average template width and height
    w, h = averageBoxes(posImages, templateScale, binSize=ppc)
    # get wahsed labels
    labels = washTxtGt(GT_PATH, curDataSet, w, h)
    print("...   {} Positive images, {} Negative images".format(len(posImages), len(negImages)))
    print("... Done\n")

    # Preparing feature vectors
    print('... Preparing training data')
    # Calculate the feature vectors of positive data
    posFeature = calPositive(posImages, w, h, orientations, ppc, cpb)
    # Calculate teh feature vectors of negative data
    negFeature = calNegative(negImages, labels, negFeatureVectorNum, w, h, oRate, orientations, ppc, cpb)

    # Mix Positive and Negative data to get training data
    # in target:
    #       -1: positive, 1: negative
    trainData = posFeature + negFeature
    trainTarget = [-1 for i in range(len(posFeature))] + [1 for i in range(len(negFeature))]
    print("...   {} Positive feature vectors".format(len(posFeature)))
    print("...   {} Negative feature vectors".format(len(negFeature)))
    print("... Done\n")

    # Train Classifer
    tbeg = time.time()
    print("... Training SVM")
    clf = trainSVM(trainData, trainTarget, probability=probability, kernel=kernel)
    tend = time.time()
    print("...   cost {} sec".format(round(tend-tbeg)))
    print("...   feature vector length: {}, w: {}, h: {}, o: {}".format(w*h*orientations ,w, h, orientations))
    print("...   test using positive feature vectors result: {:.2f}%".format(testSVMbyPosData(clf, posFeature)*100))
    print("... Done\n")

    # save trained SVM
    print("... Saving SVM")
    svmInfo = [w, h, orientations, ppc, cpb]
    # Dump training data, used to reinforce this classifier
    with open(os.path.join(SVM_PATH, saveFileName+'.data'), 'wb') as svmData:
        pickle.dump(trainData, svmData)
    with open(os.path.join(SVM_PATH, saveFileName+'.target'), 'wb') as svmTarget:
        pickle.dump(trainTarget, svmTarget)
    with open(os.path.join(SVM_PATH, saveFileName), 'wb') as svmFile:
        pickle.dump(clf, svmFile)
    with open(os.path.join(SVM_PATH, saveFileName+'.info'), 'wb') as svmInfoFile:
        pickle.dump(svmInfo, svmInfoFile)
    print("...   File: {}".format(os.path.join(SVM_PATH, saveFileName)))
    print("... Done\n")

    return


def test(SVM_PATH, loadFileName, TEST_PATH, GT_PATH, curDataSet,\
         pLevel, pScale, wStep, pThreshold, oRate, WRONG_PATH):
    """
    Run detect method to all images inside folder TEST_PATH
    And will give an final Faliure and Success ratio
    """
    # Load in classifier
    print("... Loading in SVM data")
    with open(os.path.join(SVM_PATH, loadFileName), 'rb') as svmFile:
        clf = pickle.load(svmFile)
    with open(os.path.join(SVM_PATH, loadFileName+'.info'), 'rb') as svmInfoFile:
        svmInfo = pickle.load(svmInfoFile)
    w, h, orientations, ppc, cpb = svmInfo
    print("... Done\n")

    # testing
    print("... Testing")
    testBg = time.time()

    tt = 0
    ts = 0

    labels = washTxtGt(GT_PATH, curDataSet, w, h)
    raw_labels = washTxtGt(GT_PATH, curDataSet, w, h, sizeWash=False)

    mpResults = []
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(7)

    #for f in tqdm.tqdm(os.listdir(TEST_PATH)):
    for f in os.listdir(TEST_PATH):
        CUR_FILE = os.path.join(TEST_PATH, f)

        # Multiple process speed up
        mpResult = pool.apply_async(doTest, args=(CUR_FILE, f, clf, w, h, pLevel, pScale, wStep,\
                          pThreshold, labels, raw_labels, oRate, WRONG_PATH,\
                          orientations, ppc, cpb))
        mpResults.append(mpResult)

    pool.close()
    pool.join()

    for mpResult in mpResults:
        tot, suc = mpResult.get()
        tt = tt + tot
        ts = ts + suc
        #print('...   {}, {}'.format(tt, ts))

    testEd = time.time()

    print('Result: ')
    print('Total: {}, Correct: {}, Wrong: {}'.format(tt, ts, tt-ts))
    print('Ratio: {:.2f}'.format(ts/tt*100))
    print('Using {} sec.'.format(testEd - testBg))
    print("... Done\n")

    return


def reinforce(SVM_PATH, loadFileName, saveFileName, NEG_PATH, GT_PATH,\
              curDataSet, pLevel, pScale, wStep, pThreshold, oRate, WRONG_PATH,\
              curTime):
    """
    Reinforce training using negtive images.
    """

    # Run one time test in Negative images
    test(SVM_PATH, loadFileName, NEG_PATH, GT_PATH, curDataSet,\
             pLevel, pScale, wStep, pThreshold, oRate, WRONG_PATH)

    # Load in feature vectors
    print("... Load in feature vectors")
    with open(os.path.join(SVM_PATH, loadFileName+'.info'), 'rb') as svmInfoFile:
        svmInfo = pickle.load(svmInfoFile)
    w, h, orientations, ppc, cpb = svmInfo
    with open(os.path.join(SVM_PATH, loadFileName+'.data'), 'rb') as svmData:
        data = pickle.load(svmData)
    with open(os.path.join(SVM_PATH, loadFileName+'.target'), 'rb') as svmTarget:
        target = pickle.load(svmTarget)
    print("... Done\n")

    # Get new negtive feature vectors
    for f in os.listdir(WRONG_PATH):
        image = ski.color.rgb2gray(plt.imread(os.path.join(WRONG_PATH, f)))
        image = ski.transform.resize(image, [h, w])
        feature = hog(image, orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb))
        data.append(feature)
        target.append(1)

    # Retraining Classifire and save it
    print("... Training new Classifier")
    nclf = trainSVM(data, target, probability=True, kernel='rbf')
    print("... Done\n")

    # Saving new classifier
    print("... Saving new SVM")
    svmInfo = [w, h, orientations, ppc, cpb]
    # Dump training data, used to reinforce this classifier
    with open(os.path.join(SVM_PATH, saveFileName+'.data'), 'wb') as svmData:
        pickle.dump(data, svmData)
    with open(os.path.join(SVM_PATH, saveFileName+'.target'), 'wb') as svmTarget:
        pickle.dump(target, svmTarget)
    with open(os.path.join(SVM_PATH, saveFileName), 'wb') as svmFile:
        pickle.dump(nclf, svmFile)
    with open(os.path.join(SVM_PATH, saveFileName+'.info'), 'wb') as svmInfoFile:
        pickle.dump(svmInfo, svmInfoFile)

    with open(os.path.join(SVM_PATH, saveFileName+'[{}]'.format(curTime)), 'wb') as svmFile:
        pickle.dump(nclf, svmFile)
    with open(os.path.join(SVM_PATH, saveFileName+'[{}]'.format(curTime)+'.info'), 'wb') as svmInfoFile:
        pickle.dump(svmInfo, svmInfoFile)
    print("...   File: {}".format(os.path.join(SVM_PATH, saveFileName)))
    print("... Done\n")

    return


if __name__ == '__main__':
    # Load args from user
    args = parser.parse_args()

    # Load in configures
    if not args.c == None:
        if os.path.isfile(args.c):
            configs = init(args.c)
        else:
            print('Configure file {} dosen\'t exist'.format(args.c))
            exit()
    else:
        configs = init('C:\\Yuan\\CS216\\FinalProject\\default.config')

    # Actions
    if args.action == 'train':
        print('Training Classifier')
        train(configs['POS_PATH'], configs['NEG_PATH'],\
              configs['GT_PATH'], configs['curDataSet'], configs['templateScale'],\
              configs['orientations'], configs['ppc'], configs['cpb'],\
              configs['negFeatureVectorNum'], configs['SVM_PATH'], configs['saveFileName'],\
              configs['kernel'], configs['probability'], configs['oRate'])
    elif args.action == 'reinforce':
        #print(configs['WRONG_PATH'])
        print('Reinforcing Classifier')
        for i in range(configs['rTime']):
            print('\n... reinforce time {}'.format(i+1))
            print('... mining hard negative data.\n')
            # cLear wrong dir
            for f in os.listdir(configs['WRONG_PATH']):
                os.remove(os.path.join(configs['WRONG_PATH'], f))
            reinforce(configs['SVM_PATH'], configs['loadFileName'], configs['saveFileName'],\
                 configs['NEG_PATH'],configs['GT_PATH'], configs['curDataSet'],\
                 configs['pLevel'], configs['pScale'], configs['wStep'],\
                 configs['pThreshold'], configs['oRate'], configs['WRONG_PATH'], i)
    elif args.action == 'test':
        print('Testing Classifier')
        test(configs['SVM_PATH'], configs['loadFileName'], configs['TEST_PATH'],\
             configs['GT_PATH'], configs['curDataSet'], configs['pLevel'],\
             configs['pScale'], configs['wStep'], configs['pThreshold'],\
             configs['oRate'], configs['WRONG_PATH'])
    else:
        print('The action Parameter must be one of [train | mine | test].')
