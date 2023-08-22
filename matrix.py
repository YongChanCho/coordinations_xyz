import numpy as np
import cv2

# 체크보드의 가로, 세로 내부 코너 개수
corners_x = 5
corners_y = 4

# 체크보드 패턴의 크기 (미터 단위)
square_size = 0.025

# 체크보드 코너 좌표 생성
objp = np.zeros((corners_x * corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2) * square_size

# 촬영한 이미지 파일 로드
image = cv2.imread("img0.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체크보드 코너 찾기
ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)

if ret:
     # 코너 좌표 개선
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
     # 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners2], gray.shape[::-1], None, None)
    
     # 결과 출력
    print("카메라 매트릭스:\n", mtx)
    print("왜곡 계수:\n", dist)
    print("회전 벡터 (rvecs):\n", rvecs)
    print("변위 벡터 (tvecs):\n", tvecs)
else:
    print("체크보드 패턴을 찾을 수 없습니다.")
