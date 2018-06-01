import cv2 as cv
import numpy as np
img=cv.imread('night.jpg',1)
blur1=cv.bilateralFilter(img,5,75,75)
blur2=cv.bilateralFilter(img,9,75,75)
blur3=cv.bilateralFilter(img,5,150,150)
blur4=cv.bilateralFilter(img,9,150,150)
blur5=cv.bilateralFilter(img,5,300,300)
blur6=cv.bilateralFilter(img,9,300,300)
blur7=cv.bilateralFilter(img,5,20,20)
blur8=cv.bilateralFilter(img,9,20,20)
blur=[img,blur1,blur2,blur3,blur4,blur5,blur7,blur8]
titl=['Orig','1','2','3','4','5','6','7','8']
for i in range(8):
	cv.namedWindow(titl[i],cv.WINDOW_FULLSCREEN)
	cv.imshow(titl[i],blur[i])
cv.waitKey(0)
cv.destroyAllWindows()
