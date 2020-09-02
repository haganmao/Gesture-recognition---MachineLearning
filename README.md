# Gesture-recognition---MachineLearning
The goal is to write Python scripts to perform gesture recognition.
In order to achieve the goal, there are several steps involved.


1.Pre-Training (Using TrainingData)
2.Training (TrainingData)
3.Evaluation (SelfEvalData)



# Demo
## Model Thresholds used:
    low Hue = 0
    low_Sat = 0
    low_lum = 3

### Computing the binary image by tweaking the thresholds: These values resulted the best in getting the right edge detection and contour detection.
![d1](https://github.com/haganmao/Gesture-recognition---MachineLearning/blob/master/Picture%201.png "d1") 

<br>
<br>
<br>

### Traning the imgs with hardcode thresholds parameters

![d2](https://github.com/haganmao/Gesture-recognition---MachineLearning/blob/master/Picture%202.png "d2") 

<br>
<br>
<br>

### Self-Evaluation 

![d3](https://github.com/haganmao/Gesture-recognition---MachineLearning/blob/master/Picture%203.png "d3") 

<br>
<br>
<br>


![d4](https://github.com/haganmao/Gesture-recognition---MachineLearning/blob/master/Picture%204.png "d4") 

### This resulted in a 29/50 score.




<br>
<br>

# code demo partial

`read imgs automatically from one folder with using glob library 
```python
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
   
```
<br>
<br>


`get the matrix based on y_true,y_predict, and print the matrices
```python
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


   
```











