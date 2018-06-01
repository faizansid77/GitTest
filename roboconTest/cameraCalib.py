import numpy as np
import cv2 as cv 
import yaml
cv.namedWindow('img',cv.WINDOW_FULLSCREEN)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []
cap = cv.VideoCapture(1)
found = 0
while(found < 10):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    if (ret == True) & (cv.waitKey(1) & 0xff==ord('q')):
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        frame = cv.drawChessboardCorners(frame, (7,6), corners2, ret)
        found += 1
    cv.imshow('img', frame)

cap.release()
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)