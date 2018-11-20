#!/usr/bin/env python

import cv2
import sys
import os
import numpy as np


predictions2Label = {0: "ON", 1: "OFF"}


def getTrainTest(path, class_val, test_fraction = 0.2):
  testData = []
  trainData = []
  trainLabels = []
  testLabels = []
  inputDir = os.path.expanduser(path)

  # Get images from the directory and find number of train and test samples
  if os.path.isdir(inputDir):
    images = os.listdir(inputDir)
    images.sort()
    nTest = int(len(images) * test_fraction)

  for counter, img in enumerate(images):

    im = cv2.imread(os.path.join(inputDir, img))

    if counter < nTest:              # Add nTest samples to testing data
      testData.append(im)
      testLabels.append(class_val)
    else:                             # Add nTrain samples to training data
      trainData.append(im)
      trainLabels.append(class_val)

  return trainData, trainLabels, testData, testLabels


def computeHOG(hog, data):

  hogData = []
  for image in data:
    hogFeatures = hog.compute(image)
    hogData.append(hogFeatures) 

  return hogData

def prepareData(data):
  featureVectorLength = len(data[0])
  features = np.float32(data).reshape(-1,featureVectorLength)
  return features

def svmInit(C, gamma):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  #model.setDegree(4)
  
  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, samples, labels):
    predictions = svmPredict(model, samples)
    accuracy = (labels == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy*100))
    return accuracy

if __name__ == '__main__':

  # Path1 is class 0 and Path2 is class 1
  path1 = './data/ON2'
  path2 = './data/OFF2'

  # Initialize hog parameters
  winSize = (240, 320)
  blockSize = (8, 8)
  blockStride = (8, 8)
  cellSize = (4, 4)
  nbins = 9
  derivAperture = 0
  winSigma = 4.0
  histogramNormType = 1
  L2HysThreshold = 2.0000000000000001e-01
  gammaCorrection = 1
  nlevels = 64

  hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                          derivAperture, winSigma, histogramNormType,
                          L2HysThreshold, gammaCorrection, nlevels, 1)

  # Get training and testing images for both classes
  negTrainImages, negTrainLabels, negTestImages, negTestLabels = getTrainTest(path1, 0, .2)
  posTrainImages, posTrainLabels, posTestImages, posTestLabels = getTrainTest(path2, 1, .2)

  # Append Positive and Negative Images for Train and Test
  trainImages = np.concatenate((np.array(negTrainImages), np.array(posTrainImages)), axis=0)
  testImages = np.concatenate((np.array(negTestImages), np.array(posTestImages)), axis=0)

  # Append Positive and Negative Labels for Train and Test
  trainLabels = np.concatenate((np.array(negTrainLabels), np.array(posTrainLabels)), axis=0)
  testLabels = np.concatenate((np.array(negTestLabels), np.array(posTestLabels)), axis=0)

  ####### Feature computation for the training and testing data  ######
  trainHOG = computeHOG(hog, trainImages)
  testHOG = computeHOG(hog, testImages)
  
  # Convert hog data into features recognized by SVM model
  trainFeatures = prepareData(trainHOG)
  testFeatures = prepareData(testHOG)

  ###########  SVM Training  ##############
  model = svmInit(C=10, gamma=0.001)  # C = 0.1, gamma 10 for linear kernel
  model = svmTrain(model, trainFeatures, trainLabels)
  model.save("./results/ClassModel2.yml")


  ##########  SVM Testing  ###############
  # We will load the model again and test the model
  # This is just to explain how to load an SVM model
  # You can use the model directly
  savedModel = cv2.ml.SVM_load("./results/ClassModel2.yml")

  # Find accuracy of the test data by evaluating the model on the test data
  accuracy = svmEvaluate(savedModel, testFeatures, testLabels)

  # Perform for a separate test image
  #filename = "./mixall/1.jpg"
  #if len(sys.argv) == 2:
  #  filename = sys.argv[1]
  #cropped = cv2.imread(filename)
  #cropped = getCroppedEyeRegion(testImage)
  #testHOG = computeHOG(hog, np.array([cropped]))
  #testFeatures = prepareData(testHOG)
  #predictions = svmPredict(savedModel, testFeatures)
  #print("Prediction = {}".format(predictions2Label[int(predictions[0])]))

  #cv2.imshow("testImage", testImage)
  #cv2.imshow("cropped", cropped)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()