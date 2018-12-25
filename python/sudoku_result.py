import pandas as pd
from random import randint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import random
import sys
import cv2
import matplotlib.pylab as pl
import math


with open("sudoku_output.txt") as ansFile:
    ans = [line.split() for line in ansFile]

check = int(ans[0][0])

if(check == -1):
	print ("****************Sudoku could not be solved. Try for another Image.****************\n")
	#os.system("rm testingdata.txt")
	#os.system("rm sudoku_input.txt")
	#os.system("rm sudoku_output.txt")
	exit()


# load image
image_sudoku_original = cv2.imread('./sudoku_img/sudoku.jpg')

#gray image
image_sudoku_gray = cv2.cvtColor(image_sudoku_original,cv2.COLOR_BGR2GRAY)

#adaptive threshold
thresh = cv2.adaptiveThreshold(image_sudoku_gray,255,1,1,11,15)

# find the countours
hierarchy, contours0, hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)

# size of the image (height, width)
h, w = image_sudoku_original.shape[:2]

# copy the original image to show the posible candidate
image_sudoku_candidates = image_sudoku_original.copy()

# biggest rectangle
size_rectangle_max = 0;
for i in range(len(contours0)):
    # aproximate countours to polygons
    approximation = cv2.approxPolyDP(contours0[i], 4, True)

    # has the polygon 4 sides?
    if (not (len(approximation) == 4)):
        continue;
    # is the polygon convex ?
    if (not cv2.isContourConvex(approximation)):
        continue;
        # area of the polygon
    size_rectangle = cv2.contourArea(approximation)
    # store the biggest
    if size_rectangle > size_rectangle_max:
        size_rectangle_max = size_rectangle
        big_rectangle = approximation

#show the best candidate
approximation = big_rectangle
for i in range(len(approximation)):
    cv2.line(image_sudoku_candidates,
             (big_rectangle[(i%4)][0][0], big_rectangle[(i%4)][0][1]),
             (big_rectangle[((i+1)%4)][0][0], big_rectangle[((i+1)%4)][0][1]),
             (255, 0, 0), 2)

IMAGE_WIDHT = 28
IMAGE_HEIGHT = 28
SUDOKU_SIZE = 9
N_MIN_ACTVE_PIXELS = 10

# sort the corners to remap the image
def getOuterPoints(rcCorners):
    ar = [];
    ar.append(rcCorners[0, 0, :]);
    ar.append(rcCorners[1, 0, :]);
    ar.append(rcCorners[2, 0, :]);
    ar.append(rcCorners[3, 0, :]);

    x_sum = sum(rcCorners[x, 0, 0] for x in range(len(rcCorners))) / len(rcCorners)
    y_sum = sum(rcCorners[x, 0, 1] for x in range(len(rcCorners))) / len(rcCorners)

    def algo(v):
        return (math.atan2(v[0] - x_sum, v[1] - y_sum)
                + 2 * math.pi) % 2 * math.pi
        ar.sort(key=algo)

    return (ar[3], ar[0], ar[1], ar[2])

#point to remap
points1 = np.array([
                    np.array([0.0,0.0] ,np.float32) + np.array([252,0], np.float32),
                    np.array([0.0,0.0] ,np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([0.0,252], np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([252,252], np.float32),
                    ],np.float32)
outerPoints = getOuterPoints(approximation)
points2 = np.array(outerPoints,np.float32)

#Transformation matrix
pers = cv2.getPerspectiveTransform(points2,  points1 );

#remap the image
warp = cv2.warpPerspective(image_sudoku_original, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT));
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
finalimg = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

#imgout = cv2.imread("sudoku.jpg")
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
for i in range(0,9):
	for j in range(0,9):
		#if (valid[i][j] == 1):
		#	vy=1
		#else:
		cv2.putText(finalimg, str(ans[i][j]), (j*28+5,i*28+23), font, 0.7, (0,0,0), 2)

#cv2.putText(finalimg,'0',(28,28), font, 1,(0,0,0),2)	
cv2.imshow("Result",finalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
