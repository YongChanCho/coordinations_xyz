import numpy as np
import cv2

def drawAxis(camera_parameters, markers, frame):
    axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)
    mtx, dist = camera_parameters.camera_matrix, camera_parameters.dist_coeff

    for marker in markers:
        rvec, tvec = marker.rvec, marker.tvec
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        corners = marker.corners
        corner = tuple(corners[0].ravel())
        cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0,0,255), 2)
        cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
        cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'X', tuple(imgpts[0].ravel()), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Y', tuple(imgpts[1].ravel()), font, 0.5, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Z', tuple(imgpts[2].ravel()), font, 0.5, (255,0,0), 2, cv2.LINE_AA)



 
        

