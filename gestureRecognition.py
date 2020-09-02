import numpy as np
import cv2 as cv
import os

def readImage(digit, imageNo):

    #get current work dir 
    cwd = os.getcwd()

    #relative path join
    trainSet = os.path.join(cwd,'TrainingData')

    print('!~~~~~~~~~~~~0')
    print(cwd)
    print(trainSet)
    #hand_1_0_bot_seg_1_cropped.png
    imagePre = '/hand1_'
    imagePost = '_bot_seg_'
    imagePostPost = '_cropped.png'
    
    #for i in range(1, 4):
    imageName = trainSet + imagePre + str(digit) + imagePost + str(imageNo) + imagePostPost
    # print (imageName)
    image = cv.imread(imageName)
    # imagehsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow('Loaded image', image)
    cv.waitKey(10)
    return image
        
