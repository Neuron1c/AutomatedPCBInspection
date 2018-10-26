import numpy as np
import cv2
import test
from matplotlib import pyplot as plt
import glob

image = cv2.imread('checkerboard.jpg')
x = 9
y = 7
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('chessboardDump/9x7/*.jpg')

# print(images)

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x,y),None)
    print(corners.shape)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (x,y), corners,ret)
        # plt.imshow(img)
        # plt.show()

# cv2.destroyAllWindows()
imgpoints[:] = [x*2 for x in imgpoints]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


# print(mtx,dist)
np.save('mtx',mtx)
np.save('dist',dist)

img = cv2.imread('checker.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]

img = cv2.resize(dst,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

x = 10
y = 7

ret, corners = cv2.findChessboardCorners(gray, (x,y),None)
# print(corners[0,0])

cv2.drawChessboardCorners(img, (x,y), corners,ret)
plt.imshow(img)
plt.show()

count = 0
hyp = 0

for i in range(7):
    for j in range(9):

        count += 1
        off = i*10 + j

        x = corners[off, 0, 0] - corners[off + 1, 0, 0]
        y = corners[off, 0, 1] - corners[off + 1, 0, 1]
        
        hyp += np.sqrt(x*x + y*y)
        # print(count, hyp/count)


hyp = hyp/count

hyp = (hyp/10)*2

# np.save('hyp',hyp)
print(hyp)
# cv2.imwrite('calibresult.png',dst)
