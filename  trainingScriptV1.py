import numpy as np
import cv2 as cv
import gestureRecognition as gs
import csv
import glob
import os
import writeToFile as wf


#get current work dir 
cwd = os.getcwd()




path_input = "TrainingData//*.png"
path = os.path.join(cwd,'TrainingData//')




#initialised low hue low sat low lum, and high hue,high sat, hige lum
max_value = 255
low_Hue = 0
low_Sat = 0
low_lum = 3
max_value_Hue = 360//2
high_Hue = max_value_Hue
high_Sat = max_value
high_lum = max_value
#Model parameters end


nput_window = 'Input'
detection_window = 'Hand Detection'
total_moments = 7
total_samples = 150


#Morphological operation kernel
kernel = np.ones((5,5),np.uint8)

training_moments = np.zeros((total_moments,total_samples))

# for sample in range(1,10,1):
sample = 1
for pathName in sorted(glob.glob(path_input)):
    imageName = os.path.basename(pathName)
    # imageName = str(sample) + extension
  
    img = cv.imread(path + imageName)
    print('Loaded image named' + imageName)

    if img is None:
        print ('Unable to load image.')


    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_threshold = cv.inRange(img_HSV, (low_Hue, low_Sat, low_lum), (high_Hue, high_Sat, high_lum))
    cv.imshow('Input window', img)
    closing = cv.morphologyEx(img_threshold,cv.MORPH_CLOSE,kernel)
    cv.imshow('binary original',img_threshold)
    cv.imshow('Closing',closing)


    moments = cv.HuMoments(cv.moments(closing)).flatten()
    training_moments[:, sample-1] = moments[:]

    
    cv.waitKey(1)
    sample += 1
    


#write to file
fileName = os.path.join(cwd,'momentsTrainingData_V1.txt')
print('!~~~~~~~~~~~~~~~filename')
print(fileName)
print ('Writing to a file named' + fileName)
wf.write_moments(fileName, training_moments)

