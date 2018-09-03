import csv
import cv2
import numpy as np
import contours
from matplotlib import pyplot as plt

def componentDimensions(componentCode):
    i = []
    while componentCode:
        digit = componentCode % 10
        # do whatever with digit
        i.append(digit)
        # remove last digit from number (as integer)
        componentCode //= 10
        
    compWidth = i[1] + i[0]/10
    compHeight = i[3] + i[2]/10
    
    return compWidth, compHeight
    
def locate(img, originX, originY):
    
    
    compWidth, compHeight = componentDimensions(2012)
    
    height, width, depth = img.shape

    boardDimensionX = 90
    boardDimensionY = 50

    ratioX = width/boardDimensionX
    ratioY = height/boardDimensionY
    
    ratio = (ratioX+ratioY)/2

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
                
                
                x =  int(originX + np.round(float(row[2])*ratio))
                y =  int(originY - np.round(float(row[3])*ratio))
                
                if(float(row[4]) == 90 or float(row[4]) == 270):
                    corner1x = int(x - np.round(compWidth*ratio))
                    corner1y = int(y - np.round(compHeight*ratio))
                    
                    corner2x = int(x + np.round(compWidth*ratio))
                    corner2y = int(y + np.round(compHeight*ratio))
                
                else:
                    
                    corner1x = int(x - np.round(compHeight*ratio))
                    corner1y = int(y - np.round(compWidth*ratio))
                    
                    corner2x = int(x + np.round(compHeight*ratio))
                    corner2y = int(y + np.round(compWidth*ratio))
                    
                
                # rectangle = ((corner1x,corner1y), (corner2x,corner2y), 0)

                rectangle = img[corner1y:corner2y,corner1x:corner2x]
                
                
                imgList.append(rectangle)
                
                # img = cv2.circle(img,(x ,y),2,(0,0,255),6)
                
                # cv2.rectangle(img,(corner1x,corner1y),(corner2x,corner2y),(0,255,0),3)
                
                line_count += 1
    
    
    return imgList