import numpy as np
import cv2
import test
from matplotlib import pyplot as plt
import glob

image = cv2.imread('checkerboard.jpg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('chessboardDump/*.jpg')
print(images)
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,5),None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (5,5), corners,ret)
        plt.imshow(img)
        plt.show()

cv2.destroyAllWindows()
