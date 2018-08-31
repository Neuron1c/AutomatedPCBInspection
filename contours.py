import numpy as np
import cv2
import rotation
from matplotlib import pyplot as plt

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def cropArea(img, rect):
    
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop


widthMin = 300
heightMin = 300

img = cv2.imread('test3.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


rect = 0
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    width = rect[1][0]
    height = rect[1][1]

    if ((width >= widthMin) and (height > heightMin)):
        rectangle = rect


box = cv2.boxPoints(rectangle)
box = np.int0(box)

# cv2.drawContours(img,[box],0,(0,0,255),10)

img = cropArea(img, rectangle)

rotation.getRotation(img)