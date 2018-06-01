import cv2 as cv
import numpy as np
from imutils.video import VideoStream
cap=VideoStream(src=1).start()
cv.namedWindow('Original',cv.WINDOW_FULLSCREEN)
while True:
    frame=cap.read()
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    bw=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    bw=np.float32(bw)
    out=cv.cornerHarris(bw,2,3,0.02)
    #out=cv.dilate(out,None)
    frame[out>0.01*out.max()]=[0,255,0]
    cv.imshow('Output',frame)
cv.destroyAllWindows()