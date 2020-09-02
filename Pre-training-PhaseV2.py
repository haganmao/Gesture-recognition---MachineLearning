import numpy as np
import cv2 as cv
import gestureRecognition as gs


#model parameters starts
max_value = 255
low_Hue = 0
low_Sat = 17
low_lum = 19
max_value_Hue = 360//2
high_Hue = max_value_Hue
high_Sat = max_value
high_lum = max_value
input_window = 'Input'
detection_window = 'Hand Detection'
lowH_n = 'Low H'
lowS_n = 'Low S'
lowV_n = 'Low V'
highH_n = 'High H'
highS_n = 'High S'
highV_n = 'High V'
digit = 0
kernel = np.ones((5,5),np.uint8)
#model parameters ends


def low_Hue_trackbar(val):
    global low_Hue
    global high_Hue
    low_Hue = val
    low_Hue = min(high_Hue-1, low_Hue)
    cv.setTrackbarPos(lowH_n, detection_window, low_Hue)
    threshold_callback()

def high_Hue_trackbar(val):
    global low_Hue
    global high_Hue
    high_Hue = val
    high_Hue = max(high_Hue, low_Hue+1)
    cv.setTrackbarPos(highH_n, detection_window, high_Hue)
    threshold_callback()

def low_S_trackbar(val):
    global low_Sat
    global high_Sat
    low_Sat = val
    low_Sat = min(high_Sat-1, low_Sat)
    cv.setTrackbarPos(lowS_n, detection_window, low_Sat)
    threshold_callback()

def high_S_trackbar(val):
    global low_Sat
    global high_Sat
    high_Sat = val
    high_Sat = max(high_Sat, low_Sat+1)
    cv.setTrackbarPos(highS_n, detection_window, high_Sat)
    threshold_callback()

def low_V_trackbar(val):
    global low_lum
    global high_lum
    low_lum = val
    low_lum = min(high_lum-1, low_lum)
    cv.setTrackbarPos(lowV_n, detection_window, low_lum)
    threshold_callback()

def high_V_trackbar(val):
    global low_lum
    global high_lum
    high_lum = val
    high_lum = max(high_lum, low_lum+1)
    cv.setTrackbarPos(highV_n, detection_window, high_lum)
    threshold_callback()


#create tackbar
cv.namedWindow(input_window)
cv.namedWindow(detection_window, cv.WINDOW_AUTOSIZE  )
cv.createTrackbar(lowH_n, detection_window , low_Hue, max_value_Hue, low_Hue_trackbar)
cv.createTrackbar(highH_n, detection_window , high_Hue, max_value_Hue, high_Hue_trackbar)
cv.createTrackbar(lowS_n, detection_window , low_Sat, max_value, low_S_trackbar)
cv.createTrackbar(highS_n, detection_window , high_Sat, max_value, high_S_trackbar)
cv.createTrackbar(lowV_n, detection_window , low_lum, max_value, low_V_trackbar)
cv.createTrackbar(highV_n, detection_window , high_lum, max_value, high_V_trackbar)

#input image consisting of all details
def threshold_callback():
    key = cv.waitKey(30)
    global digit
    
    #check if n key is pressed
    if ord('n') == key & 0xFF:
        print('!~~~~~~~~~~~n key is pressed')
        digit = digit + 1
        print(digit)
        if digit == 10:
            digit = 0
    elif ord('q') == key & 0xFF:
        return False

    #read image based on 2 parameters
    img = gs.readImage(digit, 1)
   
    

    #check if img is exist
    if img is None:
        print ('Unable to load image.')
    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_threshold = cv.inRange(img_HSV, (low_Hue, low_Sat, low_lum), (high_Hue, high_Sat, high_lum))
    temp = np.zeros((200, 1000), float)
    cv.imshow(input_window, img)
    cv.imshow(detection_window, temp)
    cv.imshow('Small window output', img_threshold)


    #Morphological Operation
    erosion = cv.erode(img_threshold,kernel,iterations=1)
    dilation = cv.dilate(img_threshold,kernel,iterations=1)
    opening = cv.morphologyEx(img_threshold,cv.MORPH_OPEN,kernel)
    closing = cv.morphologyEx(img_threshold,cv.MORPH_CLOSE,kernel)

    



    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = cv.drawContours(img, contours, -1, (0,255,0), 1)
    cv.imshow('Contours', drawing)
    
    # Compute seven Hu moments
    moments = cv.HuMoments(cv.moments(img_threshold)).flatten()
    print('!~~~~~~~~~~~Hu moments '+ str(digit))
    print(type(moments))
    print(moments)

    cv.waitKey(1)
    return True


key2 = True
while key2 is True:
    key2 = threshold_callback()
# cv.waitKey(0) & 0xFF #Press escape to exit program
cv.destroyAllWindows()

