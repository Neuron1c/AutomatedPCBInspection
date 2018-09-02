import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

def locate(img, originX, originY):
    height, width, depth = img.shape

    boardDimensionX = 90
    boardDimensionY = 50

    ratioX = width/boardDimensionX
    ratioY = height/boardDimensionY
    
    ratio = (ratioX+ratioY)/2

    originX = round(originX)
    originY = round(originY)
    
    with open('pnp.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:

                line_count += 1
            else:
                

                x =  int(originX + np.round(float(row[2])*ratio))
                y =  int(originY - np.round(float(row[3])*ratio))
                
                
                img = cv2.circle(img,(x ,y),2,(0,0,255),6)
                
                line_count += 1

    return img