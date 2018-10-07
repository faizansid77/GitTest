import cv2 as cv
import numpy as np
from imutils.video import VideoStream
cap=VideoStream(src=1).start()
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()
cv.namedWindow('orginal',cv.WINDOW_FULLSCREEN)
cv.namedWindow('masked',cv.WINDOW_FULLSCREEN)
while True:
    frame=cap.read()
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    fgmask=fgbg.apply(frame)
    cv.imshow('orginal',frame)
    cv.imshow('masked',fgmask)