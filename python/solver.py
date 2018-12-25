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

f1 = open('testingdata.txt','w')
outfile = open('sudoku_input.txt', 'w')

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
size_rectangle_max = 0
for i in range(len(contours0)):
    # aproximate countours to polygons
    approximation = cv2.approxPolyDP(contours0[i], 4, True)

    # has the polygon 4 sides?
    if (not (len(approximation) == 4)):
        continue
    # is the polygon convex ?
    if (not cv2.isContourConvex(approximation)):
        continue
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

#show red border image
_=pl.imshow(image_sudoku_candidates, cmap=pl.gray())
_=pl.axis("off")
_=pl.show()

IMAGE_WIDHT = 28
IMAGE_HEIGHT = 28
SUDOKU_SIZE = 9
N_MIN_ACTVE_PIXELS = 10

# sort the corners to remap the image
def getOuterPoints(rcCorners):
    ar = []
    ar.append(rcCorners[0, 0, :])
    ar.append(rcCorners[1, 0, :])
    ar.append(rcCorners[2, 0, :])
    ar.append(rcCorners[3, 0, :])

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
pers = cv2.getPerspectiveTransform(points2,  points1 )

#remap the image
warp = cv2.warpPerspective(image_sudoku_original, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT))
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
finalimg = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

#show Undistorded image
_=pl.imshow(finalimg, cmap=pl.gray())
_=pl.axis("off")
_=pl.show()


# have to delete borderline
def extract_number(x, y):
    #square -> position x-y
    im_number = warp_gray[x*IMAGE_HEIGHT:(x+1)*IMAGE_HEIGHT][:, y*IMAGE_WIDHT:(y+1)*IMAGE_WIDHT]

    #threshold
    im_number_thresh = cv2.adaptiveThreshold(im_number,255,1,1,15,9)
    #delete active pixel in a radius (from center) 
    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = math.sqrt( (IMAGE_WIDHT/2 - i)**2  + (IMAGE_HEIGHT/2 - j)**2)
            if dist_center > 9:
                im_number_thresh[i,j] = 0

    n_active_pixels = cv2.countNonZero(im_number_thresh)
    return [im_number, im_number_thresh, n_active_pixels]

def find_biggest_bounding_box(im_number_thresh):
    hierarchy, contour, hierarchy = cv2.findContours(im_number_thresh.copy(),
                                         cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)

    biggest_bound_rect = []
    bound_rect_max_size = 0
    for i in range(len(contour)):
         bound_rect = cv2.boundingRect(contour[i])
         size_bound_rect = bound_rect[2]*bound_rect[3]
         if  size_bound_rect  > bound_rect_max_size:
             bound_rect_max_size = size_bound_rect
             biggest_bound_rect = bound_rect
    #bounding box a little more bigger
    x_b, y_b, w, h = biggest_bound_rect
    x_b= x_b-1
    y_b= y_b-1
    w = w+2
    h = h+2
                
    return [x_b, y_b, w, h]

sudoku = np.zeros(shape=(9*9,IMAGE_WIDHT*IMAGE_HEIGHT))

def Recognize_number( x, y):
    """
    Recognize the number in the rectangle
    """    
    #extract the number (small squares)
    [im_number, im_number_thresh, n_active_pixels] = extract_number(x, y)

    if n_active_pixels> N_MIN_ACTVE_PIXELS:
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        im_t = cv2.adaptiveThreshold(im_number,255,1,1,15,9)
        number = im_t[y_b:y_b+h, x_b:x_b+w]

        if number.shape[0]*number.shape[1]>0:
            number = cv2.resize(number, (IMAGE_WIDHT, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            ret,number2 = cv2.threshold(number, 127, 255, 0)
            number = number2.reshape(1, IMAGE_WIDHT*IMAGE_HEIGHT)
            sudoku[x*9+y, :] = number
            return 1

        else:
            sudoku[x*9+y, :] = np.zeros(shape=(1, IMAGE_WIDHT*IMAGE_HEIGHT));
            return 0

f,axarr= pl.subplots(9,9)
for i in range(SUDOKU_SIZE):
    for j in range(SUDOKU_SIZE):
        Recognize_number(i, j)
        axarr[i, j].imshow(cv2.resize(sudoku[i*9+j, :].reshape(IMAGE_WIDHT,IMAGE_HEIGHT), (IMAGE_WIDHT,IMAGE_HEIGHT)), cmap=pl.gray())
        axarr[i, j].axis("off")

pl.show()

# Write on testingdata.txt
valid = np.zeros((28,28))

for row_number in range(0,9):
	for col_number in range(0,9):
		cellimg = np.zeros((28,28)) 	#to store the 28 * 28 image for debugging purpose
		#running the loop to take in the pixel values for each cells. 2 pixels padding is done to ignore the cell borders.
		count_pixel=0
        
		for i in range(0,28):
			for j in range(0,28):
				#pixel_value = finalimg.item(row_number*28+i,col_number*28+j)
				pixel_value = sudoku[row_number*9+col_number][i*28+j]
				if (pixel_value == 255):
					pixel_value = 1
					count_pixel = count_pixel+1
				cellimg[i,j] = pixel_value
		if (count_pixel != 0):
			valid[row_number,col_number]=1
		for i in range(0,28):
			for j in range(0,28):
				f1.write(str(int(cellimg.item(i,j)))+" ");
		f1.write('\n')
f1.close()
		
		
# ======================= KNN ============================
print("===== Start KNN =====")
with open("trainingdata.txt") as textFile:
    features = [line.split() for line in textFile]
with open("traininglable.txt") as textFile:
    tagg = [line.split() for line in textFile]
tagi=np.array(tagg)
tag=np.ravel(tagi)
with open("testingdata.txt") as textFile:
    test = [line.split() for line in textFile]
clf = KNeighborsClassifier(n_neighbors=10,weights='distance')
clf.fit(features, tag)
preds = clf.predict(test)
k = 0
for i in range(0,9):
	for j in range(0,9):
		if (valid[i][j] == 1):
			outfile.write(str(preds[k])+" ")
		else:
			outfile.write("0 ")
		k = k+1
	outfile.write('\n')
outfile.close()

print("===== Finish KNN, Please execute C++ Sudoku_Solver =====")
#os.system("./Debug/sudoku_recognizer.exe")


