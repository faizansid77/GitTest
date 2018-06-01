import cv2 as cv
import numpy as np
img=cv.imread('sudoku.jpg',0)
img=cv.GaussianBlur(img,(5,5),0)
ret,th1=cv.threshold(img,127,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.namedWindow('Original',cv.WINDOW_NORMAL)
cv.namedWindow('Threshold',cv.WINDOW_NORMAL)
cv.imshow('Original',img)
cv.imshow('Threshold',th1)
cv.waitKey(0)
cv.destroyAllWindows()
