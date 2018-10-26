import numpy as np
import cv2
import test
from matplotlib import pyplot as plt
import glob

mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
hyp = np.load('hyp.npy')
img = cv2.imread('checker2.jpg')

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

x = 10
y = 7
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objpoints1 = [] # 3d point in real world space
imgpoints1 = []

images = glob.glob('chessboardDump/10x7/*.jpg')

# print(images)

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    img1 = cv2.undistort(img, mtx, dist, None, newcameramtx)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


    # ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # Find the chess board corners
    
    ret, corners = cv2.findChessboardCorners(gray, (x,y),None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, (x,y),None)

    # print(corners.shape)
    # print(ret)
    # If found, add object points, image points (after refining them)
    
    print(ret)
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
print("halfway")

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    # img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x,y),None)
    # print(corners.shape)
    # print(ret)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints1.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints1.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (x,y), corners,ret)


imgpoints1[:] = [x*2 for x in imgpoints1]

# print(imgpoints)

r = 0
r1 = 0
count = 0
for i in range(np.array(imgpoints[0]).shape[0]):
    count += 1
    # print(imgpoints1[0][i][0][0])

    x1 = int(imgpoints1[0][i][0][0])
    y1 = int(imgpoints1[0][i][0][1])

    x2 = int(imgpoints[0][i][0][0])
    y2 = int(imgpoints[0][i][0][1])

    r1 += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    r += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)**2
r1 = r1/count
r = np.sqrt(r/count)
print(r,r1)

# img[:,:,:] = 0

# height, width, _ = img.shape

# y_points = 40
# x_points = 20
# coordinates_distorted = np.zeros((x_points*y_points,2))
# coordinates_corrected = np.zeros((x_points*y_points,2))

# d_height = height/(y_points + 1)
# d_width = width/(x_points + 1)

# cx = mtx[0,2]
# cy = mtx[1,2]

# for i in range(x_points):
#     for j in range(y_points):
#         x = (int)((i+1)*d_width)
#         y = (int)((j+1)*d_height)

#         coordinates_distorted[i*y_points+j,0] = x 
#         coordinates_distorted[i*y_points+j,1] = y

#         img[y, x ,:] = 255

# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# count = 0

# r = 0
# cx = int(cx)
# cy = int(cy)

# coordinates_distorted = np.array([coordinates_distorted])
# pts_uv = cv2.undistortPoints(coordinates_distorted, mtx,dist)
# pts_3d = cv2.convertPointsToHomogeneous(np.float32(pts_uv))

# pts_3d = pts_3d[:,0,:].T
# pts = np.dot(mtx,pts_3d).T
# pts = pts[:,0:2]




# for i in range(x_points*y_points):

#     x1 = coordinates_distorted[0,i,0]
#     y1 = coordinates_distorted[0,i,1]


#     x2 = pts[i,0]
#     y2 = pts[i,1]

#     x1 = int(x1)
#     y1 = int(y1)
#     x2 = int(x2)
#     y2 = int(y2)

#     img[y1 - 20: y1 + 20, x1 - 20 : x1 + 20,:] = 255
#     img[y2 - 20: y2 + 20, x2 - 20 : x2 + 20,2]  = 255

#     r += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)**2

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

# r = np.sqrt(r/(x_points*y_points))
# print(r)
