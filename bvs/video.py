import numpy as np
import cv2

m = 100

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 480)  # 打开并设置摄像头

# # 左相机内参
# left_camera_matrix = np.array([[964.932662651462, -0.928935316133250, 594.416643614206],
#                                [0, 968.706014744399, 383.671384353612],
#                                [0, 0, 1]])
#
# # 左相机畸变系数:[k1, k2, p1, p2, k3]
# left_distortion = np.array([[0.169243640566226, -0.189431583258940, -1.277505526759820e-04, 0.00005, 0]])
#
# # 右相机内参
# right_camera_matrix = np.array([[964.932662651462, -0.928935316133250, 654.397150214948],
#                                 [0, 968.706014744399, 383.671384353612],
#                                 [0, 0, 1]])
# # 右相机畸变系数:[k1, k2, p1, p2, k3]
# right_distortion = np.array([[0.181177064777336, -0.229106428406259, -0.001662852101870, 0.001157038140893, 0]])
#
# # 旋转矩阵
# R = np.array([[0.999970660582154, -8.71089067540394e-05, -0.00765965971368544],
#               [8.76726222895005e-05, 0.999999993473269, 7.32596304892221e-05],
#               [0.00765965328212658, -7.39290235472568e-05, 0.999970661692680]])
#
# # 平移向量
# T = np.array([[63.6549345176799],
#               [-0.141067658206410],
#               [0.368256008611823]])

# 左相机内参
left_camera_matrix = np.array([[642.3164,-0.6483,289.8883],
                               [0, 641.9616, 257.2920],
                               [0, 0, 1]])

# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[0.1828,-0.2176,0.0011,0, 0]])

# 右相机内参
right_camera_matrix = np.array([[641.0773,-0.2869,329.8542],
                                [0, 641.1759,238.8562],
                                [0, 0, 1]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]
right_distortion = np.array([[0.1865,-0.0041,0, 0]])

# 旋转矩阵
R = np.array([[1,0,-0.0079],
              [0,1,0],
              [0.0079,0,1]])

# 平移向量
T = np.array([[66.0106],
              [-0.0810],
              [0.2552]])
size = (640, 480)

# 鼠标回调函数
def cb(e, x, y, f, p):
    global start
    global end


    if x < cv2.getTrackbarPos('disparities', 'depth') * 16 or x > size[0]:
        return
    if e == cv2.EVENT_LBUTTONDOWN:
        start = points3d[y][x]/200
        print('起始:', (round(start[0], 2), round(start[1], 2), -round(start[2], 2)))
    if e == cv2.EVENT_LBUTTONUP:
        end = points3d[y][x]/200
        print('终止:', (round(end[0], 2), round(end[1], 2), -round(end[2], 2)))
        distance = np.sqrt(np.sum((start - end) ** 2))
        print('距离:', distance, 'cm')

cv2.namedWindow('depth')
cv2.namedWindow('rectify')
cv2.createTrackbar('disparities', 'depth', 1, 5, lambda x: None)
cv2.createTrackbar('block', 'depth', 7, 31, lambda x: None)
cv2.createTrackbar('multiplier', 'depth', 1, 10000, lambda x: None)
cv2.createTrackbar('lambda', 'depth', 9000, 12000, lambda x: None)
cv2.createTrackbar('sigmaColor', 'depth', 200, 1000, lambda x: None)
cv2.setMouseCallback('rectify', cb, None)

while True:
    ret, frame = cap.read()
    if ret:
        ImageL = frame[0:480, 0:640]
        ImageR = frame[0:480, 640:1280]

        # 立体校正
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix,
                                                          right_distortion, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
                                                          alpha=0)

        # 校正映射
        left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size,
                                                           cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size,
                                                             cv2.CV_16SC2)

        # 校正图像
        ImageL_rectified = cv2.remap(ImageL, left_map1, left_map2, cv2.INTER_LINEAR)
        ImageR_rectified = cv2.remap(ImageR, right_map1, right_map2, cv2.INTER_LINEAR)

        # 拼接图像
        Image = np.concatenate((ImageL_rectified, ImageR_rectified), axis=1)
        Image[::40, :] = (0, 255, 0)
        d = cv2.getTrackbarPos('disparities', 'depth') * 16
        b = cv2.getTrackbarPos('block', 'depth')
        m = cv2.getTrackbarPos('multiplier', 'depth')
        lmd = cv2.getTrackbarPos('lambda', 'depth')
        sgmcolor = cv2.getTrackbarPos('sigmaColor', 'depth')

        matcher0 = cv2.StereoSGBM_create(0, d, b, 24 * b, 96 * b, 12, 10, 50, 32, 63, cv2.STEREO_SGBM_MODE_SGBM)
        matcher1 = cv2.ximgproc.createRightMatcher(matcher0)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher0)
        wls_filter.setLambda(lmd)
        wls_filter.setSigmaColor(sgmcolor / 100)

        disp0 = np.int16(matcher0.compute(ImageL_rectified, ImageR_rectified))
        disp1 = np.int16(matcher1.compute(ImageR_rectified, ImageL_rectified))

        depth = wls_filter.filter(disp0, ImageL_rectified, None, disp1).astype(np.float32) / 16.
        points3d = cv2.reprojectImageTo3D(depth, Q)

        frame = Image.copy()
        cv2.line(frame, (d, 0), (d, size[1]), (0, 255, 0), 1)
        cv2.imshow('rectify', frame)

        depthNorm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depthNorm = np.uint8(depthNorm)

        # 显示黑白的深度图
        cv2.imshow('depth', depthNorm)

        cv2.waitKey(100)

