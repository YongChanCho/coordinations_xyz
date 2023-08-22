import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self,folder, rows, cols, square_size = 0.079, singlePnP = True, mtx = None, dist = None):
        self.folder = folder
        self.mtx = mtx
        self.dist = dist
        self.rows = rows
        self.cols = cols
        self.square_size = square_size
        self.objp = None
        self.axis = None
        self.singlePnP = singlePnP
        self.generate_board()
      
   
    def generate_board(self):
        # Generate the 3D points of the intersections of the chessboard pattern
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.rows, 0:self.cols].T.reshape(-1, 2)
        self.objp = objp * self.square_size

        # Generate the axis vectors
        self.axis = np.float32([[self.square_size, 0, 0], [0, self.square_size, 0], [0, 0, -self.square_size]]).reshape(-1, 3)

    def estimate_pose(self, image_names):
        # Loop over all images
        for image_name in image_names:
            # Extract chessboard corners
            img = cv2.imread(self.folder + image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found_corners, corners = cv2.findChessboardCorners(gray, (self.rows, self.cols), None)
            if found_corners:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (self.rows, self.cols), (-1, -1), criteria=criteria)

                # Use solve PnP to determine the rotation and translation between camera and 3D object
                ret, rvec, tvec = cv2.solvePnP(self.objp, corners, self.mtx, self.dist)

                # Project the axis into the image
                imgpts, jac = cv2.projectPoints(2 * self.axis, rvec, tvec, self.mtx, self.dist)

                # Draw the axes
                img = self.draw_axes(img, corners, imgpts)

                # Calculate camera position. Following: https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp?rq=1
                rotM = cv2.Rodrigues(rvec)[0]
                cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)

                imgpts, jac = cv2.projectPoints(cameraPosition, rvec, tvec, self.mtx, self.dist)
                # Draw a circle in the center of the image (just as a reference) and draw a line from the top left intersection to the projected camera position
                img = self.draw(img, corners[0], imgpts[0])
                cv2.imshow('img', img)
                k = cv2.waitKey(0) & 0xFF

    def draw_axes(self, img, corners, imgpts):
        # Extract the first corner (the top left)
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))

        # Color format is BGR
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        # Iterate over the points
        for i in range(len(imgpts)):
            tmp = tuple(imgpts[i].ravel())
            tmp = (int(tmp[0]), int(tmp[1]))
            img = cv2.line(img, corner, tmp, color[i], 5)
        return img

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))
        for i in range(len(imgpts)):
            tmp = tuple(imgpts[i].ravel())
            tmp = (int(tmp[0]), int(tmp[1]))
            img = cv2.line(img, corner, tmp, (255, 255, 0), 5)
        cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), 1, (255, 255, 255), 10)
        return img

    def calibrate_camera(self, images):
        # Prepare points
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.rows, 0:self.cols].T.reshape(-1, 2)
        objp = objp * self.square_size
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        
        for img_name in images:
            full_name = self.folder + img_name
            img = cv2.imread(full_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found_corners, corners = cv2.findChessboardCorners(gray, (self.rows, self.cols), None)
            if found_corners:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8,6), corners2, found_corners)
                cv2.imshow('img', img)
                k = cv2.waitKey(0) & 0xFF
                mtx = np.array([367.47894432, 0.0, 249.3915073,
                                0.0, 367.39795727, 205.2466732,
                                0.0, 0.0, 1.0]).reshape((3, 3))
                dist = np.array([0.10653164, -0.33399435, -0.00111262, -0.00186027, 0.15269198])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], mtx, dist, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        for img_name in images:
            full_name = self.folder + img_name
            img = cv2.imread(full_name)
            cv2.imshow('img', img)
            k = cv2.waitKey(0) & 0xFF
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imshow('img', dst)
            k = cv2.waitKey(0) & 0xFF


def main():
    image_names = ["img1.png","img2.png","img3.png","img4.png","img5.png","img6.png","img7.png","img8.png","img9.png"
                   ,"img10.png","img11.png","img12.png","img13.png"]
    folder = "images/"
    size = 0.079    # Unit is 'meter'

    # Precalibrated camera information
    mtx = np.array([370.68093, 0.0, 250.27853,
                    0.0, 370.65810, 206.94584,
                    0.0, 0.0, 1.0]).reshape((3, 3))
    dist = np.array([0.11336, -0.35520, 0.00076, -0.00117, 0.18745])

    # Number of checkerboard squares per row/col
    rows = 5
    cols = 4

    camera_calibrator = CameraCalibrator(folder, rows=rows, cols=cols, square_size=size, mtx=mtx, dist=dist)
    # The following line is only for camera calibration
    #camera_calibrator.calibrate_camera(image_names)
    camera_calibrator.estimate_pose(image_names)


if __name__ == '__main__':
    main()