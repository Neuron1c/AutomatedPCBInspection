import numpy as np
import cv2
from matplotlib import pyplot as plt

def getRotation(img, printCircle):

    # LIGHT BLUE
    lower = np.array([200, 200, 170], dtype = "uint8")
    upper = np.array([255, 255, 200], dtype = "uint8")
    mask1 = cv2.inRange(img, lower, upper)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    # # PINK
    # lower = np.array([110, 70, 215], dtype = "uint8")
    # upper = np.array([210, 180, 255], dtype = "uint8")
    # mask2 = cv2.inRange(img, lower, upper)


    # # YELLOW
    # lower = np.array([0, 120, 110], dtype = "uint8")
    # upper = np.array([100, 230, 210], dtype = "uint8")
    # mask3 = cv2.inRange(img, lower, upper)

    circle1 = cv2.HoughCircles(mask1, cv2.HOUGH_GRADIENT, 1.6, 50, param1=50,param2=30,minRadius=60,maxRadius=100)
    # circle2 = cv2.HoughCircles(mask2, cv2.HOUGH_GRADIENT, 1.6, 50, param1=50,param2=30,minRadius=60,maxRadius=100)
    # circle3 = cv2.HoughCircles(mask3, cv2.HOUGH_GRADIENT, 1.6, 50, param1=50,param2=30,minRadius=60,maxRadius=100)

    # circles = np.hstack((circle1,circle2,circle3))


    if circle1 is not None and printCircle == 1:
        # circles = np.uint16(np.around(circle1))

        for i in circle1[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)


    circle1 = circle1[0][0] # LIGHT BLUE
    # circle2 = circle2[0][0] # PINK
    # circle3 = circle3[0][0] # YELLOW

    halfx = mask1.shape[1]/2
    halfy = mask1.shape[0]/2

    if(circle1[1] > halfy and circle1[0] < halfx):
        return 0, circle1[0], circle1[1]

    elif(circle1[1] < halfy and circle1[0] < halfx):
        return 1, circle1[0], circle1[1] # 90

    elif(circle1[1] < halfy and circle1[0] > halfx):
        return 2, circle1[0], circle1[1] # 180

    elif(circle1[1] > halfy and circle1[0] > halfx):
        return 3, circle1[0], circle1[1] # 270

    return -1, circle1[0], circle1[1]


# img = cv2.imread('untitled.png')
# getRotation(img)
