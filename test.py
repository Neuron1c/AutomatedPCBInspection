import numpy as np
import cv2
import scipy.signal
from matplotlib import pyplot as plt

def test1(img1,img2):   #SIMPLE COLOUR GRAB


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

            img1[width-2+i,height-2+j,2] = 0
            img1[width-2+i,height-2+j,1] = 255
            img1[width-2+i,height-2+j,0] = 0

            img2[width-2+i,height-2+j,2] = 0
            img2[width-2+i,height-2+j,1] = 255
            img2[width-2+i,height-2+j,0] = 0


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

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.pad(gray,((10,10),(10,10)),'constant',constant_values=((0, 0),(0,0)))
    
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt,0.11*cv2.arcLength(cnt,True),True)
        

        if len(approx)==4:
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]

            if ((width >= 5) and (height > 5)):
                count += 1
                # cv2.drawContours(img,[cnt],0,(255),0)

            
    # if(count > 1):

    #     img = np.concatenate((img, mask), axis=1)
    #     plt.imshow(img, cmap = 'gray')
    #     plt.show()
        
    if(count > 1):
        return 0
    else:
        return 1

def test3(img1,img2):
    testor = test1(img1,img2)

    # rbg = [0,0,0]

    # for i in range(3):
    #     colour1 = img1[:,:,i]
    #     colour2 = img2[:,:,i]

    #     colour1 = colour1 - np.mean(colour1)
    #     colour2 = colour2 - np.mean(colour2)

    #     maxInd1 = np.unravel_index(np.argmax(colour1), colour1.shape)
    #     maxInd2 = np.unravel_index(np.argmax(colour2), colour.shape)

    #     max1 = colour1[maxInd1]
    #     max2 = colour2[maxInd2]

    #     colour1 = colour1/max1
    #     colour2 = colour2/max2

    #     corr = scipy.signal.correlate2d(img1, img1)
    #     corrMax = np.unravel_index(np.argmax(corr), corr.shape)

    #     rgb[i] = corr[corrMax]

    # print(testor, rgb)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

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

    if(corr[corrMax] > 600):
        return 0
    else:
        return 1



def test4(img1,img2):

    width, height, depth = img2.shape
    width = np.round(width/2).astype(int)
    height = np.round(height/2).astype(int)

    # R = img1[width,height,2] - 20
    # B = img1[width,height,1] - 20
    # G = img1[width,height,0] - 20

    R = 0
    B = 0
    G = 0

    for i in range(5):
        for j in range(5):
            R += img2[width-2+i,height-2+j,2]
            B += img2[width-2+i,height-2+j,1]
            G += img2[width-2+i,height-2+j,0]

    R = R/25 - 20
    B = B/25 - 10
    G = G/25 - 10

    if(R < 0):
        R = 0
    if(G < 0):
        G = 0
    if(B < 0):
        B = 0

    # lower = np.array([G, B, R], dtype = "uint8")
    # upper = np.array([G+20, B+20, R+40], dtype = "uint8")
    # maskRGB = cv2.inRange(img1, lower, upper)

    lower = np.array([50, 50, 50], dtype = "uint8")
    upper = np.array([80, 100, 80], dtype = "uint8")
    maskRGB = cv2.inRange(img2, lower, upper)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        

    # H = hsv[width,height,0] - 30
    # S = hsv[width,height,1] - 10
    # V = hsv[width,height,2] - 30

    H = 0
    S = 0
    V = 0

    for i in range(5):
        for j in range(5):
            H += hsv[width-2+i,height-2+j,0]
            S += hsv[width-2+i,height-2+j,1]
            V += hsv[width-2+i,height-2+j,2]

    H = H/25 - 20
    S = S/25 - 10
    V = V/25 - 20

    if(H < 0):
        H = 0
    if(S < 0):
        S = 0
    if(V < 0):
        V = 0

    lower = np.array([180, 0, 15], dtype = "uint8")
    upper = np.array([190, 20, 40], dtype = "uint8")
    maskHSV = cv2.inRange(hsv, lower, upper)
        
    CIE = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # L = CIE[width,height,0] - 18
    # u = CIE[width,height,1] - 5
    # v = CIE[width,height,2] - 5

    L = 0
    u = 0
    v = 0

    for i in range(5):
        for j in range(5):
            L += CIE[width-2+i,height-2+j,0]
            u += CIE[width-2+i,height-2+j,1]
            v += CIE[width-2+i,height-2+j,2]

    L = L/25 - 10
    u = u/25 - 5
    v = v/25 - 5

    if(L < 0):
        L = 0
    if(u < 0):
        s = 0
    if(v < 0):
        v = 0

    lower = np.array([L, u, v], dtype = "uint8")
    upper = np.array([L+20, u+10, v+10], dtype = "uint8")
    maskCIE = cv2.inRange(CIE, lower, upper)

        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if(mask[i,j] == 255):
        #             mask[i,j] = 0
        #         else:
        #             mask[i,j] = 255
        
    maskADD = cv2.add(maskHSV,maskRGB)
    mask = maskCIE*maskADD
    plt.figure('1')
    plt.imshow(maskHSV)

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # img2[width,height] = [0,0,255]
    # plt.figure('2')
    # plt.imshow(img2)

        # img = cv2.subtract(gray2,gray1)
        # print(img)

        # plt.figure('3')
        # plt.imshow(img)

    plt.show()