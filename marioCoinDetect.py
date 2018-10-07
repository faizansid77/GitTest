import cv2 as cv
import numpy as np
img=cv.imread('mario.png',1)
tem=cv.imread('marioq.png',0)
imgg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
res=cv.matchTemplate(imgg,tem,cv.TM_CCOEFF_NORMED)
w,h=tem.shape[::-1]
threshold=0.8
loc=np.where(res >= threshold)
for pt in zip(*loc[::-1]):
	cv.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,255,0),2)
cv.namedWindow('img',cv.WINDOW_FULLSCREEN)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
