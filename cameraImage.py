import numpy as np
import cv2
import test
from matplotlib import pyplot as plt

img = cv2.imread('checker2.jpg')


mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi

dst = dst[y:y+h, x:x+w]
# img = img[y:y+h, x:x+w]

cv2.imwrite('pls.jpg', dst)

x = 10
y = 7

# img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
# dst = cv2.resize(dst,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

# gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

# ret1, corners1 = cv2.findChessboardCorners(gray1, (x,y),None)
# ret2, corners2 = cv2.findChessboardCorners(gray2, (x,y),None)

# cv2.drawChessboardCorners(img, (x,y), corners1,ret1)
# cv2.drawChessboardCorners(dst, (x,y), corners2,ret2)

# pls = np.concatenate((img,dst), axis = 1)
