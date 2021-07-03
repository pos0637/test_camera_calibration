# -*- coding:utf-8 -*-
import os
import glob
import numpy as np
import cv2

DEBUG = True
CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
# 棋盘格横轴方块交点个数
x_nums = 12 - 1
# 棋盘格纵轴方块交点个数
y_nums = 9 - 1
# 世界坐标中点距离，单位毫米
world_point_distance = 15


def main():
    data_file = f'{CURRENT_PATH}/checkerboard.npz'
    test_file = f'{CURRENT_PATH}/data/Image_20210703113347256.bmp'

    '''
    ret = calibrate(f'{CURRENT_PATH}/data', data_file)
    if not ret:
        return
    '''

    with np.load(data_file) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]

    ret, rvec, tvec, o = solvePnP(test_file, mtx, dist)
    if not ret:
        return

    drawAxis(test_file, mtx, dist, rvec, tvec, o)


def calibrate(data_path, path):
    """标定

    Args:
        data_path (string): 图像文件目录
        path (string): 内参与畸变参数文件路径

    Returns:
        bool: 是否成功
    """
    images = glob.glob(f'{data_path}/*')
    world_points_list = []
    image_points_list = []
    for image in images:
        ret, size, world_points, image_points = findChessboardCorners(image)
        if ret:
            world_points_list.append(world_points)
            image_points_list.append(image_points)

    # 计算内参，畸变矩阵，外参
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        world_points_list, image_points_list, size, None, None)
    if not ret:
        return ret

    # 保存内参，畸变矩阵
    np.savez(path, mtx=mtx, dist=dist)

    # 计算误差
    mean_error = meanError(
        world_points_list, image_points_list, mtx, dist, rvecs, tvecs)
    print(f'meanError: {mean_error}')

    return ret


def findChessboardCorners(path):
    """查找棋盘格角点

    Args:
        path (string): 图像文件路径

    Returns:
        (bool, np.array, np.array): 是否成功, 世界坐标中的角点坐标集合, 图像坐标系中的角点坐标集合
    """
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]

    # 生成标定图在世界坐标中的坐标
    world_points = np.zeros((x_nums * y_nums, 3), np.float32)
    world_points[:, :2] = world_point_distance * \
        np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)

    # 查找角点
    ret, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), None)
    if not ret:
        return ret

    # 获取更精确的角点位置
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    if [corners2]:
        corners = corners2

    if DEBUG:
        cv2.drawChessboardCorners(image, (x_nums, y_nums), corners, ret)
        cv2.namedWindow(path, cv2.WINDOW_NORMAL)
        cv2.imshow(path, image)
        cv2.waitKey(0)
        cv2.destroyWindow(path)

    return ret, size, world_points, corners


def meanError(world_points_list, image_points_list, mtx, dist, rvecs, tvecs):
    """计算误差

    Args:
        world_points_list (np.array): 世界坐标中的角点坐标集合列表
        image_points_list (np.array): 图像坐标系中的角点坐标集合列表
        mtx (np.array): 内参列表
        dist (np.array): 畸变参数列表
        rvecs (np.array): 外参旋转向量列表
        tvecs (np.array): 外参平移向量列表

    Returns:
        float: 误差
    """
    mean_error = 0
    for i in range(len(world_points_list)):
        image_position2, _ = cv2.projectPoints(
            world_points_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(
            image_points_list[i], image_position2, cv2.NORM_L2) / len(image_position2)
        mean_error += error

    return mean_error / len(image_points_list)


def solvePnP(path, mtx, dist):
    """求解位姿

    Args:
        path (string): 图像文件路径
        mtx (np.array): 内参
        dist (np.array): 畸变参数

    Returns:
        (bool, np.array, np.array, np.array): 是否成功，旋转向量，平移向量，图像坐标系中的原点坐标
    """,
    ret, _, world_points, image_points = findChessboardCorners(path)
    if not ret:
        return ret

    # 获取外参，获得的旋转矩阵是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换
    ret, rvec, tvec, _ = cv2.solvePnPRansac(
        world_points, image_points, mtx, dist)
    if not ret:
        return ret

    return ret, rvec, tvec, image_points[0]


def getMatrix(rvec, tvec):
    """获取旋转矩阵和平移矩阵组成的齐次矩阵

    Args:
        rvec (np.array): 旋转向量
        tvec (np.array): 平移向量

    Returns:
        np.array: 齐次矩阵
    """
    # 罗德里格斯变换
    rotation_m, _ = cv2.Rodrigues(rvec)
    rotation_t = np.hstack([rotation_m, tvec])
    return np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])


def drawAxis(path, mtx, dist, rvec, tvec, o):
    """绘制坐标系轴

    Args:
        path (string): 图像文件路径
        mtx (np.array): 内参
        dist (np.array): 畸变参数
        rvec (np.array): 旋转向量
        tvec (np.array): 平移向量
        o (np.array): 图像坐标系中的原点坐标
    """
    image = cv2.imread(path)
    axis = world_point_distance * \
        np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    image_points, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

    image = drawLine(image, o.ravel(), image_points[0].ravel(), (255, 0, 0), 5)
    image = drawLine(image, o.ravel(), image_points[1].ravel(), (0, 255, 0), 5)
    image = drawLine(image, o.ravel(), image_points[2].ravel(), (0, 0, 255), 5)

    if DEBUG:
        cv2.namedWindow(path, cv2.WINDOW_NORMAL)
        cv2.imshow(path, image)
        cv2.waitKey(0)
        cv2.destroyWindow(path)


def drawLine(image, pt1, pt2, color, width):
    """绘制直线

    Args:
        image (np.array): 图像
        pt1 (np.array): 坐标点
        pt2 (np.array): 坐标点
        color (tuple): 颜色
        width (int): 宽度

    Returns:
        np.array: 图像
    """
    return cv2.line(image, (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])), color, width)


if __name__ == '__main__':
    main()
