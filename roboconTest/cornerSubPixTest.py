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
    out=cv.cornerHarris(bw,2,3,0.04)
    out=cv.dilate(out,None)
    _,out=cv.threshold(out,0.01*out.max(),255,0)
    out=np.uint8(out)
    _,_,_,centroi=cv.connectedComponentsWithStats(out)
    crit=(cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_COUNT,100,0.001)
    corners=cv.cornerSubPix(bw,np.float32(centroi),(5,5),(-1,-1),crit)
    res=np.hstack((centroi, corners))
    res=np.int0(res)
    frame[res[:,1],res[:,0]]=[0,255,0]
    #frame[res[:,3],res[:,2]]=[0,0,255]
    cv.imshow('Original',frame)
cv.destroyAllWindows()