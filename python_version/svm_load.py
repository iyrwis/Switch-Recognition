#!/usr/bin/env python

import cv2
import sys
import os
import glob
import numpy as np


predictions2Label = {0: "ON", 1: "OFF"}


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


def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()


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


savedModel = cv2.ml.SVM_load("./model/ClassModel2.yml")
