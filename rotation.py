import numpy as np
import cv2
from matplotlib import pyplot as plt

def getRotation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    lower = np.array([200, 200, 100], dtype = "uint8")
    upper = np.array([255, 255, 150], dtype = "uint8")
    mask = cv2.inRange(img, lower, upper)
    
    # img = cv2.imread('dave.jpg')
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(mask,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 1
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    print(lines)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


    plt.imshow(img)
    plt.show()


    lower = np.array([20, 15, 210], dtype = "uint8")
    upper = np.array([50, 45, 255], dtype = "uint8")
    mask = cv2.inRange(img, lower, upper)

    plt.imshow(mask)
    plt.show()