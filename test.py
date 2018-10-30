# -*- coding: utf-8 -*-

import os 
import sys
import dlib
import cv2
import glob
from svm_load import *

detector = dlib.simple_object_detector("detector.svm")

current_path = os.getcwd()
test_folder = current_path + '/obj_test/'

count1 = 0
count2 = 0

for f in glob.glob(test_folder + '*.jpg'):
	print 'processing file: {}'.format(f)

	img = cv2.imread(f, cv2.IMREAD_COLOR)
	pro_img = cv2.imread(f, cv2.IMREAD_COLOR)
	ori_img = cv2.imread(f, cv2.IMREAD_COLOR)

	b, g, r = cv2.split(img)
	img2 = cv2.merge([r, g, b])
	dets = detector(img2)
	print 'number of objects detected: {}'.format(len(dets))

	for index, switch in enumerate(dets):
		print 'switch {}; left {}; top {}; right {}; bottom {}'.format(index,
						switch.left(), switch.top(), switch.right(), switch.bottom())

		left = switch.left()
		top = switch.top()
		right = switch.right()
		bottom = switch.bottom()
		cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

		cropImg = ori_img[top:bottom, left:right]
		cropImg = cv2.resize(cropImg, (240, 320), interpolation=cv2.INTER_CUBIC)
		
		filename = '/home/zxy/Desktop/lastest/obj_test/segment/' + str(count1) + '_' + str(index) + '.jpg'
		cv2.imwrite(filename, cropImg)
		cropped = cv2.imread(filename)

		testHOG = computeHOG(hog, np.array([cropped]))
  		testFeatures = prepareData(testHOG)
  		predictions = svmPredict(savedModel, testFeatures)

  		print("     Picture: " + filename )
  		print("     Prediction = {}".format(predictions2Label[int(predictions[0])]))

  		if(int(predictions[0])==0):
  			pro_img = cv2.rectangle(pro_img, (left, top), (right, bottom), (255, 0, 0), 10)

  		else:
  			pro_img = cv2.rectangle(pro_img, (left, top), (right, bottom), (0, 0, 255), 10)
 
  	cv2.imwrite('/home/zxy/Desktop/lastest/results/' + 'pro_img' + str(count1) + '.jpg', pro_img)
  	cv2.imwrite('/home/zxy/Desktop/lastest/results/' + 'original_img' + str(count2) + '.jpg', img)

	count1 += 1
	count2 += 1

