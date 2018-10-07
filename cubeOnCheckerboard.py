import cv2 as cv 
import numpy as np
import yaml
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
cv.namedWindow('out',cv.WINDOW_FULLSCREEN)
mtx = np.asarray(loadeddict.get('camera_matrix'))
dist = np.asarray(loadeddict.get('dist_coeff'))
def draw(img,corners,imgpts):
    imgpts=np.int32(imgpts).reshape(-1,2)
    img=cv.drawContours(img,[imgpts[:4]],-1,(0,255,0),-3)
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,255,0),3)
    img = cv.drawContours(img, [imgpts[4:]],-1,(255,0,255),-3)
    return img
cap=cv.VideoCapture(1)
crit=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,30,0.001)
objp=np.zeros((6*7,3),np.float32)
objp[:,:2]=np.mgrid[0:7,0:6].T.reshape(-1,2)
axis=np.float32([[0,0,0],[0,3,0],[3,3,0],[3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]]).reshape(-1,3)
while True:
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret,corners=cv.findChessboardCorners(gray,(7,6),None)
    if ret==True:
        corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1),crit)
        ret,rvecs,tvecs=cv.solvePnP(objp,corners2,mtx,dist)
        imgpts,jac= cv.projectPoints(axis,rvecs,tvecs,mtx,dist)
        frame=draw(frame,corners2,imgpts)
    cv.imshow('out',frame)
    if cv.waitKey(1) & 0xff ==ord('q'):
        break
cap.release()
cv.destroyAllWindows()