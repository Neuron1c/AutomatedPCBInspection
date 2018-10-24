import scipy.signal
import numpy as np
import cv2
import csv
import test
import directoryFinder
from numpy.fft import fft2, ifft2
from timeit import default_timer as timer
from matplotlib import pyplot as plt

from model import Net
#
import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.utils.data as utils
# import torch.nn as nn
# import torch.nn.functional as F

def calculate(imgList1, imgList2):

    model = Net()
    model.load_state_dict(torch.load('convNet.een'))
    model.eval()


    with open('pnp.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:

                line_count += 1
            else:
                if line_count == 1 :
                    rotation = [row[4]]
                    code = [row[0]]
                    present = [row[7]]
                    line_count += 1
                else:
                    rotation.append(row[4])
                    code.append(row[0])
                    present.append(row[7])
                    line_count += 1

    print('Code','\tTest1', '\tTest2', '\tTest3', '\tTest6','\tTest7')
    # len(imgList1)
    mypath = 'trainingSet/2/'
    count = directoryFinder.getLastCount(mypath)

    for i in range(len(imgList1)):

        original1 = imgList1[i]
        original2 = imgList2[i]

        img1 = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        img2 = cv2.cvtColor(original2, cv2.COLOR_BGR2GRAY).astype(np.float64)

        gray1 = img1 - img1.mean()
        gray2 = img2 - img2.mean()

        # start = timer()

        fft1 = np.pad(gray1,((0,gray2.shape[0]-1),(0,gray2.shape[1]-1)),'constant',constant_values=((0, 0),(0,0)))
        fft2 = np.pad(gray2,((0,gray1.shape[0]-1),(0,gray1.shape[1]-1)),'constant',constant_values=((0, 0),(0,0)))

        fft1 = np.fft.fft2(fft1)
        fft2 = np.conjugate(np.fft.fft2(fft2))

        corr = np.real(np.fft.ifft2(fft1*fft2))
        corr = np.roll(corr, (corr.shape[0] - 1)//2, axis = 0)
        corr = np.roll(corr, (corr.shape[1] - 1)//2, axis = 1)   

        # corr = scipy.signal.correlate2d(gray1, gray2)

        ind = np.unravel_index(np.argmax(corr), corr.shape)

        # plt.imshow(np.concatenate((corr1,corr), axis = 1))
        # plt.show()
        # plt.imshow(np.concatenate((gray1,gray2), axis = 1))
        # plt.show()


        ind1 = ind[1]
        ind0 = ind[0]

        x1 = corr.shape[1] - ind1 - gray1.shape[1]
        y1 = corr.shape[0] - ind0 - gray1.shape[0]

        x2 = corr.shape[1] - ind1
        y2 = corr.shape[0] - ind0

        x3 = gray1.shape[1] - (corr.shape[1] - ind1)
        y3 = gray1.shape[0] - (corr.shape[0] - ind0)

        x4 = x3 + gray2.shape[1]
        y4 = y3 + gray2.shape[0]

        if(x1 < 0):
            x1 = 0
        if(y1 < 0):
            y1 = 0

        if(x2 > gray2.shape[1]):
            x2 = gray2.shape[1]
        if(y2 > gray2.shape[0]):
            y2 = gray2.shape[0]

        if(x3 < 0):
            x3 = 0
        if(y3 < 0):
            y3 = 0

        if(x4 > gray1.shape[1]):
            x4 = gray1.shape[1]
        if(y4 > gray1.shape[0]):
            y4 = gray1.shape[0]


        img1 = img1[y3:y4 , x3:x4]
        img2 = img2[y1:y2 , x1:x2]

        original1 = original1[y3:y4 , x3:x4, :]
        original2 = original2[y1:y2 , x1:x2, :]

        img1 = np.uint8(img1)
        img2 = np.uint8(img2)

        # x = np.concatenate(original1,original2)
        # plt.imshow(x)
        # plt.show()
        #
        temp = original2[:]

        if(int(rotation[i]) == 90 or int(rotation[i]) == 270):
            temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # if(code[i][0] == 'C' and present[i] == "YES"):
        #     print(count)
        #     location = mypath + str(count) + ".jpg"
        #     cv2.imwrite(location, temp)
        #     count += 1

        
        # print(code[i],'\t',test.test1(original1,original2), '\t',test.test2(original1,original2),'\t',test.test3(original1,original2),'\t',test.test6(original1,original2),'\t',test.test7(temp,code[i],model))
        print(present[i],test.test1(original1,original2))
        
        # if present[i] == 'NO':

        
        # test.testRotation(original1,original2,'I')
