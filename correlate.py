import scipy.signal
import numpy as np
import cv2
import test
from matplotlib import pyplot as plt

def calculate(imgList1, imgList2):
    for i in range(len(imgList1)):

        original1 = imgList1[i]
        original2 = imgList2[i]

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

        print(test.test1(original1,original2))
        
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

