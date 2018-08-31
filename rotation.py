import numpy as np
import cv2
from matplotlib import pyplot as plt



def getRotation(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # LIGHT BLUE
    lower = np.array([150, 150, 50], dtype = "uint8")
    upper = np.array([230, 220, 140], dtype = "uint8")
    mask1 = cv2.inRange(img, lower, upper)

    #  PINK
    lower = np.array([100, 60, 210], dtype = "uint8")
    upper = np.array([210, 180, 255], dtype = "uint8")
    mask2 = cv2.inRange(img, lower, upper)

    # YELLOW
    lower = np.array([0, 120, 110], dtype = "uint8")
    upper = np.array([100, 230, 210], dtype = "uint8")
    mask3 = cv2.inRange(img, lower, upper)
    
    circle1 = cv2.HoughCircles(mask1, cv2.HOUGH_GRADIENT, 1.6, 50,  param1=50,param2=30,minRadius=0,maxRadius=0)
    circle2 = cv2.HoughCircles(mask2, cv2.HOUGH_GRADIENT, 1.6, 50,  param1=50,param2=30,minRadius=0,maxRadius=0)
    circle3 = cv2.HoughCircles(mask3, cv2.HOUGH_GRADIENT, 1.6, 50,  param1=50,param2=30,minRadius=0,maxRadius=0)


    circles = np.hstack((circle1,circle2,circle3))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)


    circle1 = circle1[0][0]
    circle2 = circle2[0][0]
    circle3 = circle3[0][0]


    print(circle1)
    print(circle2)
    print(circle3)

    if(circle1[0] < circle3[0] and circle1[1] > circle2[1]):
        return 0

    elif(circle1[0] < circle2[0] and circle1[1] < circle3[1]):
        return 1 # 90

    elif(circle1[0] > circle3[0] and circle1[1] < circle2[1]):
        return 2 # 180

    elif(circle1[0] > circle2[0] and circle1[1] > circle3[1]):
        return 3 # 270

    return -1


# img = cv2.imread('untitled.png')
# getRotation(img)