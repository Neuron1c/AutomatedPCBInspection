import numpy as np
import cv2
from matplotlib import pyplot as plt

def test1(img1,img2):

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


    print(R1,R2)
    print(B1,B2)
    print(G1,G2)

    if(R2 < R1 + 40 and R2 > R1 - 40):
        if(B2 < B1 + 20 and B2 > B1 - 20):
            if(G2 < G1 + 30 and G2 > G1 - 30):
                return 1
    
    
    return 0

    
        

def test2(img1,img2):

    width, height, depth = img1.shape
    width = np.round(width/2).astype(int)
    height = np.round(height/2).astype(int)

    R = img1[width,height,2] - 20
    B = img1[width,height,1] - 20
    G = img1[width,height,0] - 20

    if(R < 0):
        R = 0
    if(G < 0):
        G = 0
    if(B < 0):
        B = 0

    lower = np.array([G, B, R], dtype = "uint8")
    upper = np.array([G+40, B+40, R+40], dtype = "uint8")
    maskRGB = cv2.inRange(img1, lower, upper)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        

    H = hsv[width,height,0] - 30
    S = hsv[width,height,1] - 10
    V = hsv[width,height,2] - 30
        
    if(H < 0):
        H = 0
    if(S < 0):
        S = 0
    if(V < 0):
        V = 0

    lower = np.array([H, S, V], dtype = "uint8")
    upper = np.array([H+60, S+20, V+60], dtype = "uint8")
    maskHSV = cv2.inRange(hsv, lower, upper)
        
    CIE = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)

    L = CIE[width,height,0] - 18
    u = CIE[width,height,1] - 5
    v = CIE[width,height,2] - 5
        
    if(L < 0):
        L = 0
    if(u < 0):
        s = 0
    if(v < 0):
        v = 0

    lower = np.array([L, u, v], dtype = "uint8")
    upper = np.array([L+26, u+10, v+10], dtype = "uint8")
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
    plt.imshow(mask)

        # original2[width,height] = [0,0,255]
    plt.figure('2')
    plt.imshow(img1)

        # img = cv2.subtract(gray2,gray1)
        # print(img)
        # plt.figure('3')
        # plt.imshow(img)

    plt.show()