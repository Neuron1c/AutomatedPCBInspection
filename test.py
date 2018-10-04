import numpy as np
import cv2
import scipy.signal
import random as rng
from matplotlib import pyplot as plt

def test1(img1,img2): #SIMPLE COLOUR GRAB USING HSV

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    width, height, depth = img1.shape
    width = np.round(width/2).astype(int)
    height = np.round(height/2).astype(int)

    # R1 = img1[width,height,2]
    # B1 = img1[width,height,1] 
    # G1 = img1[width,height,0]

    R1 = 0
    B1 = 0
    G1 = 0

    R2 = 0
    B2 = 0
    G2 = 0

    for i in range(5):
        for j in range(5):
            R1 += img1[width-2+i,height-2+j,2]
            B1 += img1[width-2+i,height-2+j,1]
            G1 += img1[width-2+i,height-2+j,0]

            R2 += img2[width-2+i,height-2+j,2]
            B2 += img2[width-2+i,height-2+j,1]
            G2 += img2[width-2+i,height-2+j,0]


    # R2 = img2[width,height,2]
    # B2 = img2[width,height,1]
    # G2 = img2[width,height,0]

    R1 = R1/25
    R2 = R2/25
    
    B1 = B1/25
    B2 = B2/25

    G1 = G1/25
    G2 = G2/25

    # print(np.round(R1 - R2))
    # print(np.round(B1 - B2))
    # print(np.round(G1 - G2))

    if(R2 < R1 + 50 and R2 > R1 - 50):
        if(B2 < B1 + 30 and B2 > B1 - 30):
            if(G2 < G1 + 15 and G2 > G1 - 15):
                return 0
    
    
    return 1

def test2(img1,img2): #ATTEMPT TO DETECT SQUARE SOLDER PADS

    lower = np.array([120, 120, 120], dtype = "uint8")
    upper = np.array([255, 255, 255], dtype = "uint8")
    mask = cv2.inRange(img2, lower, upper)

    mask = np.pad(mask,((10,10),(10,10)),'constant',constant_values=((0, 0),(0,0)))

    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = img2
    count = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.pad(gray,((10,10),(10,10)),'constant',constant_values=((0, 0),(0,0)))
    
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt,0.12*cv2.arcLength(cnt,True),True)
        

        if len(approx)==4:
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]

            if ((width >= 5) and (height > 5)):
                count += 1
                cv2.drawContours(img,[cnt],0,(255),0)

            
    # if(count > 1):

    #     img = np.concatenate((img, mask), axis=1)
    #     plt.imshow(img, cmap = 'gray')
    #     plt.show()
        
    if(count > 1):
        return 0
    else:
        return 1

def test3(img1,img2): #CORRELATION TEST

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    x,y = img1.shape
    area = x*y

    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)

    maxInd1 = np.unravel_index(np.argmax(img1), img1.shape)
    maxInd2 = np.unravel_index(np.argmax(img2), img2.shape)

    minInd1 = np.unravel_index(np.argmin(img1), img1.shape)
    minInd2 = np.unravel_index(np.argmin(img2), img2.shape)

    max1 = img1[maxInd1]
    min1 = img1[minInd1]

    max2 = img2[maxInd2]
    min2 = img2[minInd2]

    if(max1 > -1*min1):
        img1 = img1/max1
    else:
        img1 = img1/(-1*min1)

    if(max2 > -1*min2):
        img2 = img2/max2
    else:
        img2 = img2/(-1*min2)
           

    corr = scipy.signal.correlate2d(img1, img2)
    corrMax = np.unravel_index(np.argmax(corr), corr.shape)

    if(corr[corrMax]/area > 0.1):
        return 0
    else:
        return 1

def test4(img1,img2):

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv2.filter2D(img2, cv2.CV_32F, kernel)
    sharp = np.float32(img2)
    imgResult = sharp - imgLaplacian

    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # img2 = imgResult

    width, height, depth = img2.shape
    width = np.round(width/2).astype(int)
    height = np.round(height/2).astype(int)

    lower = np.array([50, 50, 50], dtype = "uint8")
    upper = np.array([80, 100, 80], dtype = "uint8")
    maskRGB = cv2.inRange(img2, lower, upper)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    lower = np.array([80, 0, 20], dtype = "uint8")
    upper = np.array([120, 40, 100], dtype = "uint8")
    maskHSV = cv2.inRange(hsv, lower, upper)
        
    CIE = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    lower = np.array([50, 120, 120], dtype = "uint8")
    upper = np.array([80, 130, 130], dtype = "uint8")
    maskCIE = cv2.inRange(CIE, lower, upper)

    maskADD = cv2.add(maskHSV,maskRGB)
    mask = maskCIE*maskADD


    # _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros(mask.shape).astype('uint8')
    # count = 0
    
    # for cnt in contours:

    #     approx = cv2.approxPolyDP(cnt,0.12*cv2.arcLength(cnt,True),True)  

    #     if len(approx)==4:
    #         rect = cv2.minAreaRect(cnt)
    #         width = rect[1][0]
    #         height = rect[1][1]

    #         if ((width >= 5) and (height > 5)):
    #             count += 1
    #             cv2.drawContours(mask,[cnt],0,(255),-1)

    # plt.imshow(mask)
    # plt.show()

    dist = cv2.distanceTransform(maskRGB, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1)

    dist_8u = dist.astype('uint8')

    _, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = np.zeros(dist.shape, dtype=np.int32)


    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)

    markers = cv2.watershed(imgResult, markers)

    mark = markers.astype('uint8')
    mark = cv2.bitwise_not(mark)

    colors = []
    for contour in contours:
        colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]


    plt.figure('1')
    plt.imshow(img2)

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # plt.figure('2')
    # plt.imshow(img2)

    plt.show()