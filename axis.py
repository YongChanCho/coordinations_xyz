import numpy as np
import cv2

# Load camera matrix and distortion coefficients obtained from calibration
mtx = np.array([[466.46370836, 0, 325.19957854], [0, 471.43761911, 272.60137191], [0, 0, 1]])
dist = np.array([-6.69635054e-02,6.97766724e-01, 1.8186555506e-05,-3.73539488e-03, 1.44291301e+00])
rvecs=np.array([[-0.20463966],
       [ 0.02620935],
       [-0.02738775]])
tvecs=np.array([[-0.04481435],
       [-0.04593455],
       [ 0.23555025]])
x=0.0
y=0.0
z=0.0
# 3D point in the world coordinate system
point_3d = np.array([[x, y, z]])

# Project 3D point onto the image plane
point_2d, _ = cv2.projectPoints(point_3d, rvecs, tvecs, mtx, dist)

# Draw the projected point on the image
image = cv2.imread("img0.png")
cv2.circle(image,(300,300) , 5, (0, 0, 255), -1)
cv2.circle(image,(200,300) , 5, (0, 0, 255), -1)
cv2.circle(image,(200,200) , 5, (0, 0, 255), -1)

#tuple(point_2d[0].ravel())
# Display the image with the projected point
cv2.imshow("Image with Projected Point", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
