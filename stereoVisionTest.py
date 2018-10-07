import cv2 as cv
import numpy as np 
from imutils.video import VideoStream
import imutils
left=VideoStream(src=1).start()
right=VideoStream(src=2).start()
cv.namedWindow('Left',cv.WINDOW_FULLSCREEN)
cv.namedWindow('Right',cv.WINDOW_FULLSCREEN)
cv.namedWindow('Depth',cv.WINDOW_FULLSCREEN)
stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
while True:
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    fl=left.read()
    fl=imutils.rotate_bound(fl,90)
    fr=right.read()
    fr=imutils.rotate_bound(fr,90)
    bfl=cv.cvtColor(fl,cv.COLOR_BGR2GRAY)
    bfr=cv.cvtColor(fr,cv.COLOR_BGR2GRAY)
    dis=stereo.compute(bfl,bfr)
    cv.imshow('Left',fl)
    cv.imshow('Right',fr)
    cv.imshow('Depth',dis)
cv.destroyAllWindows()