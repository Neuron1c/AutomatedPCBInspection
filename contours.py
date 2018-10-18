import numpy as np
import cv2
import rotation
import locateComponents
import correlate
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

def main(imgName):
    widthMin = 500
    heightMin = 500

    img = cv2.imread(imgName)

    mtx = np.load('mtx.npy')
    dist = np.load('dist.npy')

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('calibresult.png',dst)

    img = dst

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # plt.imshow(np.concatenate((thresh,gray), axis = 1))
    # plt.show()

    rect = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]

        if ((width >= widthMin) and (height > heightMin)):
            rectangle = rect
            widthMin = width
            heightMin = height



    # cv2.drawContours(img,contours,-1,(0,0,255),10)

    img = cropArea(img, rectangle)

    box = cv2.boxPoints(rectangle)
    box = np.int0(box)

    x, originX, originY = rotation.getRotation(img, 0)

    for i in range(x):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    x, originX, originY = rotation.getRotation(img, 1)
    # plt.imshow(img)
    # plt.show()
    imgList = locateComponents.locate(img, originX, originY)
    i = 0

    return imgList

    # for img in imgList:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.figure(i)
    #     plt.imshow(img)
    #     i = i + 1

    # # cv2.waitkey(0)
    # plt.show()


baseImgList = main('images/Golden/1.jpg')
newImgList =  main('images/populated/6.jpg')
correlate.calculate(baseImgList, newImgList)


# import directoryFinder
# mypath = 'images/populated/'
# paths = directoryFinder.getList(mypath)

# for x in paths:
#     baseImgList = main('golden.jpg')
#     newImgList =  main(mypath + x)
#     correlate.calculate(baseImgList, newImgList)
