#coding=utf-8
import os
import numpy as np
import cv2
import tensorflow as tf

def load(filepath):
	print("loaddata")
	trainRate = 0.9

	listfile = os.listdir(filepath)
	labelNum = len(listfile)
	imgNum = 0
	trainNum = 0
	for fileName in listfile:
		currentImgs = os.listdir(filepath + fileName)
		imgNum += len(currentImgs)
		trainCount = round(trainRate * len(currentImgs))
		trainNum += int(trainCount)

	height = 32
	width = 32	
	channel = 3

	trainImgs = np.empty((trainNum, height, width, channel), dtype=np.float32)
	trainLabels = np.zeros((trainNum, labelNum),dtype=np.int16)
	testImgs = np.empty((imgNum - trainNum, height, width, channel), dtype=np.float32)
	testLabels = np.zeros((imgNum - trainNum, labelNum),dtype=np.int16)
	trainIndex = 0
	testIndex = 0
	labelIndex = 0
	for fileName in listfile:
		currentImgs = os.listdir(filepath + fileName)
		trainCount = round(trainRate * len(currentImgs))
		i = 0
		for imgName in currentImgs:
			img = cv2.imread(filepath + fileName + "/" + imgName)
			res=cv2.resize(img,(height, width),interpolation=cv2.INTER_CUBIC)
			arr = np.array(res
			if i < trainCount:
				trainImgs[trainIndex,:,:,:] = arr/256.
				trainLabels[trainIndex, labelIndex] = 1
				trainIndex += 1
			else:
				testImgs[testIndex,:,:,:] = arr/256.
				testLabels[testIndex, labelIndex] = 1
				testIndex += 1
			i += 1
		labelIndex += 1
	print("shuffle")
	trainImgs, trainLabels = shuffle(trainImgs, trainLabels)
	testImgs, testLabels = shuffle(testImgs, testLabels)	

	np.save("images_train.npy", trainImgs)
	np.save("labels_train.npy", trainLabels)
	np.save("images_test.npy", testImgs)
	np.save("labels_test.npy", testLabels)
	
	print("data saved")

	return trainImgs, trainLabels, testImgs, testLabels, labelNum

def shuffle(imgs, labels):
	num = imgs.shape[0]
	perm = np.arange(num)
	np.random.shuffle(perm)
	imgs = imgs[perm]
	labels = labels[perm]
	return imgs,labels

if __name__=="__main__":
	trainImgs, trainLabels, testImgs, testLabels, labelNum = load("/home/xubaochuan/new/ml/tensorflow/caltech101/data/")
