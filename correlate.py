import scipy.signal
import numpy as np
import cv2
from matplotlib import pyplot as plt

def calculate(imgList1, imgList2):

    original1 = imgList1[0]
    original2 = imgList2[0]

    img1 = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    img2 = cv2.cvtColor(original2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    gray1 = img1 - img1.mean()
    gray2 = img2 - img2.mean()

    corr = scipy.signal.correlate2d(gray1, gray2)
    ind = np.unravel_index(np.argmax(corr), corr.shape)

    ind1 = ind[1] 
    ind0 = ind[0] 

    x1 = corr.shape[1] - ind1 - gray1.shape[1]
    y1 = corr.shape[0] - ind0 - gray1.shape[0]

    x2 = corr.shape[1] - ind1
    y2 = corr.shape[0] - ind0

    x3 = gray1.shape[1] - (corr.shape[1] - ind1)
    y3 = gray1.shape[0] - (corr.shape[0] - ind0)

    x4 = x3 + gray2.shape[1]
    y4 = y3 + gray2.shape[0]

    if(x1 < 0):
        x1 = 0
    if(y1 < 0):
        y1 = 0

    if(x2 > gray2.shape[1]):
        x2 = gray2.shape[1]
    if(y2 > gray2.shape[0]):
        y2 = gray2.shape[0]

    if(x3 < 0):
        x3 = 0
    if(y3 < 0):
        y3 = 0

    if(x4 > gray1.shape[1]):
        x4 = gray1.shape[1]
    if(y4 > gray1.shape[0]):
        y4 = gray1.shape[0]


    img1 = img1[y3:y4 , x3:x4]
    img2 = img2[y1:y2 , x1:x2]

    original1 = original1[y3:y4 , x3:x4, :]
    original2 = original2[y1:y2 , x1:x2, :]

    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # gray = np.float32(img1)
    # dst = cv2.cornerHarris(gray,2,3,0.2)
    # original1[dst>0.01*dst.max()]=[0,0,255]
    # plt.figure('harris 1')
    # plt.imshow(original1)

    # gray = np.float32(img2)
    # dst = cv2.cornerHarris(gray,2,3,0.2)
    # original2[dst>0.01*dst.max()]=[0,0,255]
    # plt.figure('harris 2')
    # plt.imshow(original2)

    # gray1 = cv2.Canny(img1,50,100)
    # gray2 = cv2.Canny(img2,50,100)
    width, height, depth = original1.shape
    width = np.round(width/2).astype(int)
    height = np.round(height/2).astype(int)

    R = original2[width,height,2]
    B = original2[width,height,1]
    G = original2[width,height,0]

    lower = np.array([G-30, B-30, R-30], dtype = "uint8")
    upper = np.array([G+30, B+30, R+30], dtype = "uint8")
    maskRGB = cv2.inRange(original2, lower, upper)

    lower = np.array([0, 0, 0], dtype = "uint8")
    upper = np.array([100, 50, 50], dtype = "uint8")
    maskHSV = cv2.inRange(original2, lower, upper)

    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if(mask[i,j] == 255):
    #             mask[i,j] = 0
    #         else:
    #             mask[i,j] = 255
    
    maskADD = cv2.add(maskHSV,maskRGB)

    img = cv2.cvtColor(imgList2[1], cv2.COLOR_BGR2RGB)
    plt.figure('1')
    plt.imshow(maskRGB)

    # original2[width,height] = [0,0,255]
    # img = cv2.cvtColor(imgList2[1], cv2.COLOR_BGR2RGB)
    # plt.figure('2')
    # plt.imshow(original2)

    # img = cv2.subtract(gray2,gray1)
    # print(img)
    # plt.figure('3')
    # plt.imshow(img)

    plt.show()