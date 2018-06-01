import cv2 as cv
import numpy as np
img=cv.imread('face.png',0)
img=cv.Canny((cv.GaussianBlur(img,(11,11),0)),100,150)
cv.namedWindow('img',cv.WINDOW_FULLSCREEN)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
