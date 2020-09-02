import numpy as np
import cv2 as cv
import gestureRecognition as gs
import csv
import glob
import os
import writeToFile as wf
import readFromFile as rf
from sklearn.metrics import confusion_matrix

#get current work dir 
cwd = os.getcwd()



#path configuration
path_input = "SelfEvalData//*.png"
path = os.path.join(cwd,'SelfEvalData//')




#initialised low hue low sat low lum, and high hue,high sat, hige lum
#model parameters starts
low_Hue = 0
low_Sat = 17
low_lum = 19
max_value = 255
max_value_Hue = 360//2
high_Hue = max_value_Hue
high_Sat = max_value
high_lum = max_value
#Model parameters end


nput_window = 'Input'
detection_window = 'Hand Detection'
total_moments = 7
total_samples = 50
training_samples = 150
training_samples_per_dight = 15


#Morphological operation kernel
kernel = np.ones((5,5),np.uint8)

# read the moments from training data text file
fileName = os.path.join(cwd,'momentsTrainingData_V2.txt')

training_moments = rf.read_moments(fileName)




#arrays for computing matrix and distances
#initialized the y_predict and y_true arrays
#initailzed the distance array
y_predict = np.zeros((total_samples),int)
y_true = np.zeros((total_samples),int)
distance = np.zeros((training_samples),float)






# for sample in range(1,10,1):
sample = 1
digit = 0
for pathName in sorted(glob.glob(path_input)):
    imageName = os.path.basename(pathName)
    # imageName = str(sample) + extension
  
    img = cv.imread(path + imageName)
    print('Loaded image named' + imageName)

    if img is None:
        print ('Unable to load image.')
        break

    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_threshold = cv.inRange(img_HSV, (low_Hue, low_Sat, low_lum), (high_Hue, high_Sat, high_lum))
    cv.imshow('Input window', img)
    closing = cv.morphologyEx(img_threshold,cv.MORPH_CLOSE,kernel)
    cv.imshow('binary original',img_threshold)
    cv.imshow('Closing',closing)
    moments = cv.HuMoments(cv.moments(closing)).flatten()


    #compute the distance with each of the training sample
    for i in range(0, training_samples):
        distance[i] = cv.norm(moments,training_moments[:, i], cv.NORM_L2)
    
    #tuple of (index and datatype)
    matching_tuples = np.where(distance == distance.min()) 
    matching_indices = matching_tuples[0]
    first_match = matching_indices[0]
    matching_digit = (first_match + 1 )//training_samples_per_dight
    y_predict[sample-1]= matching_digit
    y_true[sample-1] = digit

    print('Digit is ', digit,'Matched with ',matching_digit)
    cv.waitKey(1)


    #application for the next iteration
    if sample % 5 == 0:
        digit +=1 
    sample += 1


matrix = confusion_matrix(y_true,y_predict)
print(matrix)

