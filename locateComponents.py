import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

def componentDimensions(componentCode):
    i = []
    while componentCode:
        digit = componentCode % 10
        i.append(digit)
        componentCode //= 10

    compWidth = i[1] + i[0]/10
    compHeight = i[3] + i[2]/10

    return compWidth, compHeight

def locate(img, originX, originY):

    height, width, depth = img.shape

    # boardDimensionX = 90
    # boardDimensionY = 50

    boardDimensionX = 68
    boardDimensionY = 57

    ratioX = width/boardDimensionX
    ratioY = height/boardDimensionY

    ratio = (ratioX+ratioY)/2

    ratio = np.load('hyp.npy')

    originX = round(originX)
    originY = round(originY)

    imgList = []

    with open('pnp.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:

                line_count += 1
            else:

                if(row[0][:1] == 'R' or row[0][:1] == 'C'):
                    compWidth, compHeight = componentDimensions(int(row[5]))
                    # print(compWidth, compHeight, ratio)

                elif(row[0][:1] == 'I'):
                    compWidth = float(row[5])
                    compHeight = float(row[6])
                    # print(compWidth, compHeight, ratio)

                x =  int(originX + np.round(float(row[2])*ratio))
                y =  int(originY - np.round(float(row[3])*ratio))

                compWidth = compWidth/2
                compHeight = compHeight/2

                if(float(row[4]) == 90 or float(row[4]) == 270):

                    corner1x = int(x - np.round(compWidth*ratio + ratio*1.1))
                    corner1y = int(y - np.round(compHeight*ratio + ratio*1.1))

                    corner2x = int(x + np.round(compWidth*ratio + ratio*1.1))
                    corner2y = int(y + np.round(compHeight*ratio + ratio*1.1))

                else:

                    corner1x = int(x - np.round(compHeight*ratio + ratio*1.1))
                    corner1y = int(y - np.round(compWidth*ratio + ratio*1.1))

                    corner2x = int(x + np.round(compHeight*ratio + ratio*1.1))
                    corner2y = int(y + np.round(compWidth*ratio + ratio*1.1))

                rectangle = img[corner1y:corner2y,corner1x:corner2x]

                imgList.append(rectangle)

                # img = cv2.circle(img,(x ,y),2,(0,0,255),6)

                # cv2.rectangle(img,(corner1x,corner1y),(corner2x,corner2y),(0,255,0),3)

                line_count += 1

    # plt.imshow(img)
    # plt.show()
    return imgList
