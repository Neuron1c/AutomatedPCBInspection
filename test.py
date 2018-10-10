import numpy as np
import cv2
import scipy.signal
import random as rng
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

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

    lower = np.array([60, 60, 60], dtype = "uint8")
    upper = np.array([100, 110, 100], dtype = "uint8")
    maskRGB = cv2.inRange(img2, lower, upper)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    lower = np.array([80, 0, 20], dtype = "uint8")
    upper = np.array([120, 40, 100], dtype = "uint8")
    maskHSV = cv2.inRange(hsv, lower, upper)

    CIE = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)


    lower = np.array([55, 120, 120], dtype = "uint8")
    upper = np.array([100, 135, 135], dtype = "uint8")
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



    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1)

    dist_8u = dist.astype('uint8')

    _, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = np.zeros(dist.shape, dtype=np.int32)

    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)

    markers = cv2.watershed(img2, markers)

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

    imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
    out = np.concatenate((maskRGB,maskHSV,maskCIE,mask), axis = 1)
    plt.imshow(maskCIE)

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # plt.figure('2')
    # plt.imshow(img2)

    plt.show()

def test5(img1,img2):

    RGB1 = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    RGB2 = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]

    y1,x1,_ = img1.shape
    y2,x2,_ = img2.shape

    R = img1[:,:,2]
    G = img1[:,:,1]
    B = img1[:,:,0]

    RGB1[0][0] = np.sum(R < 63)
    RGB1[0][1] = np.sum(np.logical_and((R >= 63),(R < 127)))
    RGB1[0][2] = np.sum(np.logical_and((R >= 127),(R < 191)))
    RGB1[0][3] = np.sum(R >= 191)

    RGB1[1][0] = np.sum(B < 63)
    RGB1[1][1] = np.sum(np.logical_and((B >= 63),(B < 127)))
    RGB1[1][2] = np.sum(np.logical_and((B >= 127),(B < 191)))
    RGB1[1][3] = np.sum(B >= 191)

    RGB1[2][0] = np.sum(G < 63)
    RGB1[2][1] = np.sum(np.logical_and((G >= 63),(G < 127)))
    RGB1[2][2] = np.sum(np.logical_and((G >= 127),(G < 191)))
    RGB1[2][3] = np.sum(G >= 191)

    R = img2[:,:,2]
    G = img2[:,:,1]
    B = img2[:,:,0]

    RGB2[0][0] = np.sum(R < 63)
    RGB2[0][1] = np.sum(np.logical_and((R >= 63),(R < 127)))
    RGB2[0][2] = np.sum(np.logical_and((R >= 127),(R < 191)))
    RGB2[0][3] = np.sum(R >= 191)

    RGB2[1][0] = np.sum(B < 63)
    RGB2[1][1] = np.sum(np.logical_and((B >= 63),(B < 127)))
    RGB2[1][2] = np.sum(np.logical_and((B >= 127),(B < 191)))
    RGB2[1][3] = np.sum(B >= 191)

    RGB2[2][0] = np.sum(G < 63)
    RGB2[2][1] = np.sum(np.logical_and((G >= 63),(G < 127)))
    RGB2[2][2] = np.sum(np.logical_and((G >= 127),(G < 191)))
    RGB2[2][3] = np.sum(G >= 191)

    RGB1 = np.array(RGB1)/(x1*y1)
    RGB2 = np.array(RGB2)/(x2*y2)
    # print(RGB1[0])
    # print(RGB2[0])
    # print()
    # x = np.zeros(20)

    # vals = np.linspace(0.17,0.2,20)

    # for i in range(20):
    #     x[i] = np.allclose(RGB1,RGB2, 0 ,vals[i])

    # print(x)

    if(np.allclose(RGB1,RGB2, 0 ,0.18)):
        return 0
    return 1

def test6(img1,img2):

    lower = np.array([140, 140, 140], dtype = "uint8")
    upper = np.array([255, 255, 255], dtype = "uint8")
    mask = cv2.inRange(img1, lower, upper)

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv2.filter2D(img1, cv2.CV_32F, kernel)
    sharp = np.float32(img1)
    img1 = sharp - imgLaplacian

    img1 = np.clip(img1, 0, 255)
    img1 = img1.astype('uint8')


    imgLaplacian = cv2.filter2D(img2, cv2.CV_32F, kernel)
    sharp = np.float32(img2)
    img2 = sharp - imgLaplacian

    img2 = np.clip(img2, 0, 255)
    img2 = img2.astype('uint8')

    mask2 = np.pad(mask,((10,10),(10,10)),'constant',constant_values=((0, 0),(0,0)))

    image, contours, hierarchy = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    indSquare = None
    count = 0
    flag = 1

    for cnt in contours:

        approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        # cv2.drawContours(mask2,approx,-1,[125],3)
        if len(approx)==4:
            count += 1
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]

            if ((width >= 2) and (height > 2)):
                # cv2.drawContours(img1,[x - 10 for x in cnt],-1,(0,0,255),3)
                if flag:
                    square = approx[:]
                    indSquare = approx[:]
                    indSquare = [indSquare]
                    flag = 0
                else:
                    # print(square.shape, count)
                    square = np.concatenate((square, approx))
                    # indSquare = np.stack((indSquare, approx), axis = 1)
                    indSquare.append(approx)

                # count += 1
    # print(indSquare, count)
    # for i in range(count):


    # contours[:] = [x - 10 for x in contours]
    # cv2.drawContours(img1,contours,-1,(0,0,255), 3)
    if(indSquare == None):
        return 0

    for sqr in indSquare:
        rect = cv2.minAreaRect(sqr[:,0,:])
        box = cv2.boxPoints(rect)
        box = np.int0(box - 10)
        cv2.drawContours(mask, [box], 0,(255),-1)

    rect = cv2.minAreaRect(square)
    box = cv2.boxPoints(rect)
    box = np.int0(box-10)
    mask3 = np.array(mask)

    # cv2.drawContours(mask,[box1],0,(255),-1)
    # cv2.drawContours(mask,[box2],0,(255),-1)
    cv2.drawContours(mask3,[box],0,(255),-1)

    mask = 255 - mask

    green1 = img1[:,:,1].astype(int)
    green2 = img2[:,:,1].astype(int)


    height, width = green1.shape
    count = 0
    mean1 = 0
    mean2 = 0
    for i in range(height):
        for j in range(width):

            if(mask3[i,j] == 0):
                count += 1
                mean1 += green1[i,j]
                mean2 += green2[i,j]

    mean1 = mean1/count
    mean2 = mean2/count
    # print(mean1,mean2)

    # mean1 = np.mean(green1)
    # mean2 = np.mean(green2)

    green2 = green2.astype(int) + (mean1 - mean2)
    green2 = np.clip(green2,0,255).astype('uint8')
    green1 = green1.astype('uint8')
    mask = mask.astype('uint8')

    out = cv2.subtract(green2,green1, mask = mask)
    out = np.round(out).astype('uint8')


    mask = 255 - mask
    mask4 = np.array(mask3.astype(int) + mask.astype(int))
    mask4 = np.clip(mask4,0,255) - mask.astype(int)

    # x = np.clip(out.astype(int) + mask4.astype(int), 0,255)


    count = 0
    mean = 0

    for i in range(height):
        for j in range(width):

            if(mask4[i,j] != 0):
                count += 1
                mean += out[i,j]
            else:
                out[i,j] = 0

    mean = mean/count
    # print(mean)
    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # out = np.concatenate((gray,mask,mask4,out), axis = 1)
    # plt.imshow(out,cmap='gray')
    # plt.show()

    if(mean > 30):
        return 1
    return 0

def testRotation(img1,img2,componentCode):


    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    #
    # imgLaplacian = cv2.filter2D(img2, cv2.CV_32F, kernel)
    # sharp = np.float32(img2)
    # img2 = sharp - imgLaplacian
    #
    # img2 = np.clip(img2, 0, 255)
    # img2 = img2.astype('uint8')


    # CIE = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    #
    #
    # lower = np.array([100, 125, 125], dtype = "uint8")
    # upper = np.array([160, 130, 130], dtype = "uint8")
    # maskCIE = cv2.inRange(CIE, lower, upper)


    out = np.concatenate((img1,img2), axis = 1)
    # print(CIE[45, 77,:])

    # circle = cv2.HoughCircles(maskCIE, cv2.HOUGH_GRADIENT, 0.5, 20, param1=50,param2=10,minRadius=0,maxRadius=15)
    # print(circle)
    # for i in circle[0,:]:
    #     # draw the outer circle
    #     cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv2.circle(img2,(i[0],i[1]),2,(0,0,255),3)

    plt.imshow(img2)
    plt.show()
