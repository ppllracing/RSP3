import itertools
import pickle
import json
import sys
import warnings
import carla
import math
import numpy as np
import os
import logging
import time
import torch
import bezier
import joblib
from joblib import Memory
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from numba import njit

from .config import Configuration

path_folder_parking_learning_A_star = os.path.join(*(['/'] + os.path.dirname(__file__).split('/')[1:-3] + ['parking_learning_A_star']))

# 调用“parking_learning_A_star”下的算法
sys.path.append(path_folder_parking_learning_A_star)
from collision_check import collision_check
from map import costmap_for_bev
from planner import Planner

mem = Memory(location='./.cache', verbose=0)

# 获取相机内参
def get_camera_intrinsic(h: int, w: int, fov: float):
    # 计算相机的焦距
    focal = w / (2 * math.tan(math.radians(fov) / 2))

    # 创建一个3x3的单位矩阵
    intrinsic = np.identity(3)

    # 赋值
    intrinsic[0, 0] = intrinsic[1, 1] = focal
    intrinsic[0, 2] = w / 2
    intrinsic[1, 2] = h / 2

    return intrinsic

# 获取相机外参
def get_camera_extrinsic(transform_relative):
    # 对于rsu，transform_relative的原点是map_bev的中心点
    # 对于obu，transform_relative的原点是ego_vehicle的中心点
    # 此处参考E2EParking项目的代码实现
    cam2pixel = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )
    extrinsic = cam2pixel @ np.array(transform_relative.get_inverse_matrix())
    return extrinsic

def cal_trans(_xyzPYR=None, location=None, rotation=None):
    _trans = carla.Transform(
        carla.Location(*_xyzPYR[0:3]) if location is None else location,
        carla.Rotation(*_xyzPYR[3:]) if rotation is None else rotation
    )
    return _trans

def np2str(_np, _width=10):
    return ','.join(f'{d:>{_width}.4f}' for d in _np)

def image2np(image):
    # 图像变换
    image_np = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    h, w, fov = image.height, image.width, image.fov
    image_np = np.reshape(image_np, (h, w, 4))
    image_np = image_np[:, :, :3]
    image_np = image_np[:, :, ::-1]
    image_np = image_np.transpose(2, 0, 1)

    return image_np

# 初始化logger
def init_logger(cfg: Configuration, logger_name):
    path_logger = os.path.join(cfg.path_logs, f'{logger_name}.log')

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(path_logger)
        file_handler.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()  
        stream_handler.setLevel(cfg.logger_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

def cal_crop_start_end(ori, aim):
    start = 0
    jump = math.floor((ori - start) / (aim - 1))
    end = start + (aim - 1) * jump
    adjust = (ori - end) / 2
    start += adjust
    end += adjust
    return int(start), int(jump), int(end)

def crop_image(image, size_aim, method='direct', alignment=[0, 0]):
    h_aim, w_aim = size_aim
    h_ori, w_ori = image.shape[-2:]
    if method == 'direct':
        h_start = (h_ori - h_aim) // 2
        w_start = (w_ori - w_aim) // 2
        image_crop = image[..., h_start:h_start+h_aim, w_start:w_start+w_aim].copy()
    elif method == 'max':
        h_start, h_jump, h_end = cal_crop_start_end(h_ori, h_aim)
        w_start, w_jump, w_end = cal_crop_start_end(w_ori, w_aim)
        image_crop = image[..., h_start:h_end+h_jump:h_jump, w_start:w_end+w_jump:w_jump].copy()
    elif method == 'linspace':
        h_start = 0
        w_start = 0
        h_ids = np.linspace(h_start, h_ori-1, h_aim).round().astype(int)
        w_ids = np.linspace(w_start, w_ori-1, w_aim).round().astype(int)
        image_crop = image[:, h_ids, :][:, :, w_ids]
    else:
        assert False, 'Crop Method Error'
    
    assert image_crop.shape[-2:] == (h_aim, w_aim), 'Crop Error'

    scaling = max(h_aim/h_ori, w_aim/w_ori)

    return image_crop, (scaling, h_start, w_start)

def cal_post_tran_rot(factors, dtype):
    scaling, h_start, w_start = factors
    _post_tran = np.zeros(2, dtype=dtype)
    _post_rot = np.eye(2, dtype=dtype)

    _post_tran -= np.array([w_start, h_start], dtype=dtype)
    _post_rot *= scaling

    post_tran = np.zeros(3, dtype=dtype)
    post_rot = np.eye(3, dtype=dtype)

    post_tran[:2] = _post_tran
    post_rot[:2, :2] = _post_rot

    return post_tran, post_rot

def matrix_to_transform(matrix: np.ndarray):
    # 提取位置
    location = carla.Location(
        x=matrix[0, 3],
        y=matrix[1, 3],
        z=matrix[2, 3]
    )
    
    # 提取旋转矩阵并转换为欧拉角
    rotation_matrix = matrix[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    
    # 创建carla.Rotation (注意CARLA使用度)
    rotation = carla.Rotation(
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw)
    )
    
    return carla.Transform(location, rotation)

def get_relative_matrix(transform_a, transform_b):
    matrix_inv_a = np.array(transform_a.get_inverse_matrix())
    matrix_b = np.array(transform_b.get_matrix())
    relative_matrix = matrix_inv_a @ matrix_b
    return relative_matrix

def move_transform_by_relative_matrix(transform_a, relative_matrix):
    # 将transform_a转换为矩阵形式
    matrix_a = np.array(transform_a.get_matrix())
    
    # 计算新的变换矩阵
    matrix_b = matrix_a @ relative_matrix

    transform_b = matrix_to_transform(matrix_b)
    return transform_b

# 在bev_point_infos中选择ref_point_info覆盖的点，并将其置为set_value，其余位置为abs(1-set_value)
def select_bev_points(map_bev_layer, bev_point_infos, ref_point_info, set_value=1):
    ref_box = ref_point_info['box']
    if 'transform' in ref_point_info.keys():
        ref_transform = ref_point_info['transform']
    else:
        ref_transform = carla.Transform()

    # 检测bev的每个位置中心点是否在ref的box内，z方向上只要有一个在，就可以判定
    for i in range(bev_point_infos.shape[0]):
        for j in range(bev_point_infos.shape[1]):
            for k in range(bev_point_infos.shape[2]):
                bev_point_info = bev_point_infos[i, j, k]

                # 判断当前bev_point是否在ref内
                if ref_box.contains(bev_point_info['location'], ref_transform):
                    map_bev_layer[i, j] = set_value
                    break
                else:
                    map_bev_layer[i, j] = abs(1 - set_value)
    
    return map_bev_layer

# 创建一个上下文管理器，用于屏蔽输出
class suppress_output:
    def __enter__(self):
        self._original_stdout = sys.stdout  # 保存原始的stdout
        sys.stdout = open(os.devnull, 'w')  # 重定向stdout到null设备

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # 关闭重定向的文件
        sys.stdout = self._original_stdout  # 恢复原始stdout

# 创建一个上下文管理器，用于屏蔽警告
class suppress_warnings:
    def __enter__(self):
        self._original_warning_filters = warnings.filters[:]  # 保存当前的warning过滤器
        warnings.filterwarnings("ignore")  # 暂时忽略所有警告

    def __exit__(self, exc_type, exc_value, traceback):
        warnings.filters.clear()  # 清空当前的warning过滤器
        warnings.filters.extend(self._original_warning_filters)  # 恢复原来的warning过滤器

# 创建一个上下文管理器，用于屏蔽输出、错误和警告
class suppress_output_and_warnings:
    def __enter__(self):
        self._original_stdout = sys.stdout  # 保存原始的stdout
        self._original_stderr = sys.stderr  # 保存原始的stderr
        sys.stdout = open(os.devnull, 'w')  # 重定向stdout到null设备
        sys.stderr = open(os.devnull, 'w')  # 重定向stderr到null设备
        self._original_warning_filters = warnings.filters[:]  # 保存当前的warning过滤器
        warnings.filterwarnings("ignore")  # 暂时忽略所有警告

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # 关闭重定向的文件
        sys.stderr.close()  # 关闭重定向的错误输出文件
        sys.stdout = self._original_stdout  # 恢复原始stdout
        sys.stderr = self._original_stderr  # 恢复原始stderr
        warnings.filters.clear()  # 清空当前的warning过滤器
        warnings.filters.extend(self._original_warning_filters)  # 恢复原来的warning过滤器

# 从车辆中心点计算后轴中心的location
def cal_rear_axle_location(center_transform, wheel_base):
    rear_axle_location = carla.Location(
        x=center_transform.location.x - wheel_base / 2 * center_transform.get_forward_vector().x,
        y=center_transform.location.y - wheel_base / 2 * center_transform.get_forward_vector().y,
        z=0.0
    )
    return rear_axle_location

# 从后轴中心计算车辆中心点的location
def cal_center_location(rear_axle_transform, wheel_base):
    center_location = carla.Location(
        x=rear_axle_transform.location.x + wheel_base / 2 * rear_axle_transform.get_forward_vector().x,
        y=rear_axle_transform.location.y + wheel_base / 2 * rear_axle_transform.get_forward_vector().y,
        z=0.0
    )
    return center_location

# 按照顺序获得bbox的vertices，顺序是左前，右前，右后，左后
def get_bbox_vertices_ordered(bbox, transform):
    # 'rear left down', 'rear left up', 'rear right down', 'rear right up',
    # 'front left down', 'front left up', 'front right down', 'front right up'
    vertices = [[v.x, v.y, v.z] for v in bbox.get_world_vertices(transform)]
    vertices_up = [vertices[i] for i in [5, 7, 3, 1]]
    vertices_down = [vertices[i] for i in [4, 6, 2, 0]]
    return vertices_up, vertices_down

# 在plot中绘制标准矩形
def plot_standard_rectangle(vertices: list, ax, line='b-', inner_point=None):
    if len(vertices) == 4:
        points = vertices
    elif len(vertices) == 8:
        points = np.array(vertices)
        points = points[points[:, 2] > points[:, 2].mean()]  # 提取顶部的点
        # 按照顺时针进行排序
        center = np.mean(points, axis=0)[0:2]
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        # 按照角度进行排序，使用argsort得到索引
        sorted_indices = np.argsort(angles)
        # 按照排序的索引重新排列点
        points = points[sorted_indices]
        points = points[:, 0:2].tolist()
    # 解压缩点的坐标
    x, y, _ = zip(*points)
    x += (x[0],)
    y += (y[0],)
    ax.plot(y, x, line)

    if not inner_point is None:
        for x, y, _ in points[2:]:
            ax.plot([inner_point[1], y], [inner_point[0], x], line)

# 将carla坐标系下的xyt转换为正常坐标系下的xyt
def convert_xyt_from_carla_to_normal_coord(xyt: list):
    if len(xyt) == 3:
        x_c, y_c, t_c = xyt
        # xy对调，并调整t的起始轴
        x_n, y_n = y_c, x_c
        t_n = t_c - 90.0
        # 将t的旋转方向调整为逆时针为正，并转为弧度制
        t_n = -t_n
        t_n = math.radians(t_n)
        return [x_n, y_n, t_n]
    elif len(xyt) == 2:
        x_c, y_c = xyt
        return [y_c, x_c]
    else:
        assert False, 'xyt的长度必须为2或3'

# # 将正常坐标系下的xyt转换为carla坐标系下的xyt
# def convert_xyt_from_normal_to_carla_coord(xyt: list):
#     if len(xyt) == 3:
#         x_n, y_n, t_n = xyt
#         # xy对调，并调整t的起始轴
#         x_c, y_c = y_n, x_n
#         t_c = t_n + 90.0
#         # 将t的旋转方向调整为顺时针为正，并转为角度制
#         t_c = -t_c
#         t_c = math.radians(t_c)
#         return [x_c, y_c, t_c]
#     elif len(xyt) == 2:
#         x_n, y_n = xyt
#         return [y_n, x_n]
#     else:
#         assert False, 'xyt的长度必须为2或3'

# 将任意弧度旋转至[-pi, pi]范围内
def wrap_to_2pi(radian):
    radian_np = np.array(radian)
    radian_np_wrap = radian_np.copy()
    radian_np_wrap = (radian_np_wrap + np.pi) % (2 * np.pi) - np.pi
    assert np.sum(np.abs(np.sin(radian_np_wrap) - np.sin(radian_np))) < 1e-6 and np.sum(np.abs(np.cos(radian_np_wrap) - np.cos(radian_np))) < 1e-6, 'wrap_to_2pi error'
    
    if isinstance(radian, float):
        return radian_np_wrap.item()
    elif isinstance(radian, list):
        return radian_np_wrap.tolist()
    else:
        return radian_np_wrap

# 计算两个点的前后性
def check_0_to_1(xyt_0, xyt_1, returm_yaw=True):
    # 确保维度是[3, 1]
    xyt_0_np = np.array(xyt_0).reshape(3, 1)
    xyt_1_np = np.array(xyt_1).reshape(3, 1)

    # xyt_1与xyt_0的连线，与xyt_0方向的夹角
    yaw = wrap_to_2pi(math.atan2(xyt_1_np[1] - xyt_0_np[1], xyt_1_np[0] - xyt_0_np[0]) - xyt_0_np[2])

    flag = (-math.pi / 2.0) < yaw < (math.pi / 2.0)

    if returm_yaw:
        return flag, yaw
    else:
        return flag

# 正向坐标变换
def trans_xyt(xyt, tranform_list):
    rotate_rad, rotate_matrix, move_vector = tranform_list
    xy = rotate_matrix @ (xyt[:2] + move_vector)
    t = wrap_to_2pi(xyt[2:] + rotate_rad)
    return np.concatenate([xy, t], axis=0)
# 逆向坐标变换（与上一个函数对应，不建议单独使用）
def trans_xyt_inv(xyt, tranform_list):
    rotate_rad, rotate_matrix, move_vector = tranform_list
    xy = np.linalg.inv(rotate_matrix) @ (xyt[:2]) - move_vector
    t = wrap_to_2pi(xyt[2:] - rotate_rad)
    return np.concatenate([xy, t], axis=0)

# 计算整个序列的点与点之间的距离
def cal_distance_between_points(xy):
    xy = np.array(xy)
    diff = np.diff(xy, axis=0)
    distance = np.array([np.sqrt(np.sum(d**2)) for d in diff])
    return distance

# 通过变换将两个点的线率进行平均
def transform_average_slope(xyt):
    # 将[*, 3]转为[3, *]
    if xyt.shape[1] == 3:
        xyt = xyt.T

    # 将坐标轴原点移动至xyt_0上，并将xyt_0的方向与x轴重合
    move_vector_1 = - np.array([xyt[0, 0], xyt[1, 0]]).reshape(-1, 1)
    rotate_rad_1 = - xyt[2, 0]
    rotate_matrix_1 = np.array([
        [math.cos(rotate_rad_1), -math.sin(rotate_rad_1)],
        [math.sin(rotate_rad_1), math.cos(rotate_rad_1)]
    ])
    tranform_list_1 = [rotate_rad_1, rotate_matrix_1, move_vector_1]
    xyt_trans_ = trans_xyt(xyt, tranform_list_1)

    # 将xyt_tans_的矢量方向进行平均
    move_vector_2 = - np.array([0.0, 0.0]).reshape(-1, 1)
    rotate_rad_2 = - xyt_trans_[2, -1] / 2
    rotate_matrix_2 = np.array([
        [math.cos(rotate_rad_2), -math.sin(rotate_rad_2)],
        [math.sin(rotate_rad_2), math.cos(rotate_rad_2)]
    ])
    tranform_list_2 = [rotate_rad_2, rotate_matrix_2, move_vector_2]
    xyt_trans = trans_xyt(xyt_trans_, tranform_list_2)

    # 此时始末点指向的应该都是x轴正半轴
    assert (-math.pi / 2 < xyt_trans_[2]).all() and (xyt_trans_[2] < math.pi / 2).all(), 'Trans error'

    # 通过x的大小来判断先后
    flag_swap = False
    if xyt_trans[0, 0] > xyt_trans[0, -1]:
        xyt_trans = xyt_trans[:, ::-1]
        flag_swap = True
    
    # 将[3, *]转为[*, 3]
    xyt_trans = xyt_trans.T

    return xyt_trans, flag_swap, tranform_list_1, tranform_list_2

# 将经过平均的点进行反向处理
def transform_average_slope_inv(xyt, flag_swap, tranform_list_1, tranform_list_2):
    # 若在一开始的始末坐标点对换了，则需要在此处再次对换
    if flag_swap:
        xyt = xyt[::-1]

    # 将插值点还原至原坐标系
    xyt = xyt.T
    xyt = trans_xyt_inv(xyt, tranform_list_2)
    xyt = trans_xyt_inv(xyt, tranform_list_1)
    xyt = xyt.T

    return xyt

# 判断点的距离是否小于阈值
def check_distance_threshold(xy_0, xy_1, threshold=1e-6):
    distance = np.linalg.norm(xy_0[:2] - xy_1[:2])
    return distance < threshold

# 判断两个角度很相近小于阈值
def check_angle_threshold(theta1, theta2, threshold=1e-6):
    diff = abs(wrap_to_2pi(theta1 - theta2))
    return diff <= threshold

# 根据始末斜率
def interp_Bezier(xyt_0, xyt_1):
    xyt_0 = np.array(xyt_0)
    xyt_1 = np.array(xyt_1)

    if np.linalg.norm(xyt_0[:2] - xyt_1[:2]) < 0.1:
        return np.array([xyt_0, xyt_1])

    # 判断是否前后交换
    flag_swap = False
    if check_0_to_1(xyt_0, xyt_1, returm_yaw=False):
        pass
    else:
        xyt_0, xyt_1 = xyt_1, xyt_0
        flag_swap = True

    # 分离始末坐标和斜率
    xy_0 = xyt_0[:2]
    xy_1 = xyt_1[:2]
    t_0 = xyt_0[2]  # 弧度
    t_1 = xyt_1[2]  # 弧度

    # 生成三次贝塞尔曲线控制点
    d = np.linalg.norm(xy_1 - xy_0) / 3
    b1 = xy_0 + d * np.array([np.cos(t_0), np.sin(t_0)])
    b2 = xy_1 - d * np.array([np.cos(t_1), np.sin(t_1)])
    nodes = np.stack([xy_0, b1, b2, xy_1], axis=1)
    curve = bezier.Curve(nodes, degree=3)

    # 采样点和对应的导数
    samples = np.linspace(0, 1, 10)
    bezier_points = curve.evaluate_multi(samples).T
    # 计算每个点的斜率
    t = []
    for sample in samples:
        t_ = curve.evaluate_hodograph(sample)
        t_ = np.arctan2(t_[1], t_[0]).item()
        # 调整朝向，使每个点与初始点的方向对齐
        if len(t) == 0 or check_angle_threshold(t[-1], t_, np.pi/2):
            t.append(t_)
        else:
            t.append(wrap_to_2pi(t_ + np.pi))
    delta_t = t_0 - t[0]  # 获取初始点存在的偏差
    t = np.array(t) + delta_t  # 修正每个点的航向

    # 将坐标和航向合并
    xyt_oup = np.concatenate([bezier_points, t.reshape(-1, 1)], axis=1)

    # 检查始末点的坐标和航向是否一致，来判断计算过程有没有问题
    assert check_distance_threshold(xyt_oup[0, :2], xy_0), f'xy_0 error, {xyt_oup[0, :2]}!= {xy_0}, [{xyt_0} -> {xyt_1}]'
    assert check_distance_threshold(xyt_oup[-1, :2], xy_1), f'xy_1 error, {xyt_oup[-1, :2]}!= {xy_1}, [{xyt_0} -> {xyt_1}]'
    assert check_angle_threshold(xyt_oup[0, 2], t_0, 1e-4), f't_0 error, {xyt_oup[0, 2]}!= {t_0}, [{xyt_0} -> {xyt_1}]'
    assert check_angle_threshold(xyt_oup[-1, 2], t_1, 1e-4), f't_1 error, {xyt_oup[-1, 2]}!= {t_1}, [{xyt_0} -> {xyt_1}]'

    # 若没问题，则直接赋值始末点
    xyt_oup[0], xyt_oup[-1] = xyt_0, xyt_1

    if flag_swap:
        xyt_oup = xyt_oup[::-1]

    return xyt_oup

# 控制每个while循环的运行帧率，用于替代pygame的clock
class FPSCountroller:
    def __init__(self, fps=30):
        self.fps = fps
        self.last_time = time.time()
    
    def tick(self):
        current_time = time.time()
        delta_time = current_time - self.last_time
        if delta_time < 1.0 / self.fps:
            time.sleep(1.0 / self.fps - delta_time)
        self.last_time = current_time
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

# 将数据保存
def save_datas_to_disk(datas, path_datas_folder, name, mode='json'):
    assert mode in ['json', 'pkl'], 'Unsupported mode'
    if f'{name}.{mode}' in path_datas_folder.split('/'):
        path_datas = path_datas_folder
    else:
        path_datas = os.path.join(path_datas_folder, f'{name}.{mode}')
    if mode == 'json':
        with open(path_datas, 'w') as f:
            json.dump(datas, f, indent=1)
    elif mode == 'pkl':
        with open(path_datas, 'wb') as f:
        #     pickle.dump(datas, f)
            joblib.dump(datas, f)
    else:
        raise ValueError('Unsupported mode')
    
    while not os.path.exists(path_datas):
        print(f'Waiting for {path_datas} to be saved...')
        time.sleep(0.01)

# 读取数据
@mem.cache
def read_datas_from_disk(path_datas_folder=None, name=None, mode='json', path_datas=None):
    return read_datas_from_disk_without_cached(path_datas_folder, name, mode, path_datas)
    # assert mode in ['json', 'pkl'], 'Unsupported mode'
    # if path_datas is None:
    #     if f'{name}.{mode}' in path_datas_folder.split('/'):
    #         path_datas = path_datas_folder
    #     else:
    #         path_datas = os.path.join(path_datas_folder, f'{name}.{mode}')

    # if mode == 'json':
    #     assert True, 'json not supported now'
    #     # with open(path_datas, 'r') as f:
    #     #     datas = json.load(f)
    # else:
    #     with open(path_datas, 'rb') as f:
    #         # datas = pickle.load(f)
    #         # datas = joblib.load(f, mmap_mode='r')
    #         duration, datas = timefunc(joblib.load, path_datas)
    #         print(f'Loading {path_datas} took {duration:.2f}s')
    # return datas

# 不带缓存地读取数据
def read_datas_from_disk_without_cached(path_datas_folder=None, name=None, mode='json', path_datas=None):
    assert mode in ['json', 'pkl'], 'Unsupported mode'
    if path_datas is None:
        if f'{name}.{mode}' in path_datas_folder.split('/'):
            path_datas = path_datas_folder
        else:
            path_datas = os.path.join(path_datas_folder, f'{name}.{mode}')

    if mode == 'json':
        assert True, 'json not supported now'
        # with open(path_datas, 'r') as f:
        #     datas = json.load(f)
    else:
        with open(path_datas, 'rb') as f:
            # datas = pickle.load(f)
            # datas = joblib.load(f, mmap_mode='r')
            duration, datas = timefunc(joblib.load, path_datas)
            print(f'Loading {path_datas} took {duration:.2f}s')
    return datas

# 从二值图中获取ego的中心位置
def get_position_id_from_bev(ego_bev):
    if isinstance(ego_bev, np.ndarray):
        ego_bev = np.array(ego_bev)
        ego_position = np.mean(
            np.nonzero(ego_bev),
            axis=1,
            keepdims=True
        ).T
        if np.isnan(ego_position).any():
            ego_position = np.ones_like(ego_position) * -1.0  # 对于尚未成型的ego_bev，是无法计算出位置的，用-1代替
        ego_position_id = np.round(ego_position)
    elif isinstance(ego_bev, torch.Tensor):
        ego_position = torch.mean(
            torch.nonzero(ego_bev).to(ego_bev.dtype),
            dim=0, keepdim=True
        )
        if torch.isnan(ego_position).any():
            ego_position = torch.ones_like(ego_position) * -1.0  # 对于尚未成型的ego_bev，是无法计算出位置的，用-1代替
        ego_position_id = torch.round(ego_position)
    else:
        raise TypeError('Unsupported type')
    return ego_position_id

# 将xy坐标生成一个token
def generate_token_from_xy(xy_id, flag_cat=False):
    assert xy_id.shape[-1] == 2 and len(xy_id.shape) == 2, 'xy_id shape error'

    x_id, y_id = xy_id[:, 0], xy_id[:, 1]
    x_token = (x_id + 2).reshape(-1, 1)
    y_token = (y_id + 2).reshape(-1, 1)

    if flag_cat:
        if isinstance(x_token, np.ndarray):
            return np.concatenate([x_token, y_token], axis=1)
        elif isinstance(x_token, torch.Tensor):
            return torch.cat([x_token, y_token], dim=1)
        else:
            raise TypeError('Unsupported type')
    else:
        return x_token, y_token

# 从token中获取xy的id
def get_xy_id_from_token(xy_token, flag_cat=False):
    if isinstance(xy_token, list):
        xy_token = np.array(xy_token)
    assert xy_token.shape[-1] == 2 and len(xy_token.shape) == 2, 'xy_token shape error'

    x_token, y_token = xy_token[:, 0], xy_token[:, 1]
    x_id = x_token - 2
    y_id = y_token - 2

    if flag_cat:
        if isinstance(x_id, np.ndarray):
            return np.concatenate([x_id.reshape(-1, 1), y_id.reshape(-1, 1)], axis=1)
        elif isinstance(x_id, torch.Tensor):
            return torch.cat([x_id.reshape(-1, 1), y_id.reshape(-1, 1)], dim=1)
        else:
            raise TypeError('Unsupported type')
    else:
        return x_id.reshape(-1, 1), y_id.reshape(-1, 1)

# 将xy的id转换为全局坐标
def get_global_xy_from_xy_id(xy_ids, global_origin, resolution, flag_cat=False):
    if isinstance(xy_ids, list):
        xy_ids = np.array(xy_ids)
    assert xy_ids.shape[-1] == 2 and len(xy_ids.shape) == 2, 'xy_token shape error'

    x_o, y_o = global_origin

    # 通过id补充坐标
    x_ids, y_ids = xy_ids[:, 0], xy_ids[:, 1]
    x_ = x_o + x_ids * resolution[0] * -1.0
    y_ = y_o + y_ids * resolution[1]
    # global_path_ = np.stack([x_ids, y_ids, x_, y_], axis=1)

    if flag_cat:
        if isinstance(x_, np.ndarray):
            return np.concatenate([x_.reshape(-1, 1), y_.reshape(-1, 1)], axis=1)
        elif isinstance(x_, torch.Tensor):
            return torch.cat([x_.reshape(-1, 1), y_.reshape(-1, 1)], dim=1)
        else:
            raise TypeError('Unsupported type')
    else:
        return x_.reshape(-1, 1), y_.reshape(-1, 1)

def generate_search_matrix_and_weight_matrix(yaw, radius, angle):
    radius_ego, radius_search = radius
    center = np.array([radius_search, radius_search])
    # 通过坐标id和偏航计算代表车辆的两个圆心
    ego_2_circle_center = np.stack([
        center + radius_ego / 2 * np.array([math.cos(yaw), math.sin(yaw)]),
        center - radius_ego / 2 * np.array([math.cos(yaw), math.sin(yaw)])
    ], axis=0)
    search_matrix = np.zeros((2 * radius_search + 1, 2 * radius_search + 1))
    weight_matrix = np.zeros_like(search_matrix)
    for i in range(2 * radius_search + 1):
        for j in range(2 * radius_search + 1):
            distance_to_center = np.linalg.norm(np.array([i, j]) - center)
            distance_to_ego = min(
                np.linalg.norm(np.array([i, j]) - ego_2_circle_center[0]),
                np.linalg.norm(np.array([i, j]) - ego_2_circle_center[1])
            )
            yaw_ = math.atan2(j - center[1], i - center[0])
            if distance_to_center <= radius_search and distance_to_ego >= radius_ego and abs(wrap_to_2pi(yaw_ - yaw)) < angle / 2:
                search_matrix[i, j] = 1.0
                weight_matrix[i, j] = 1.0 - (distance_to_center - radius_ego) / (radius_search - radius_ego)
    weight_matrix /= weight_matrix.sum()
    return search_matrix, weight_matrix

# 计算每个路点的危险值
def cal_risk_degrees_for_path_points(path_points, map_bev, resolution, radius, angle):
    # 计算每个路点的危险度
    # 每个路点的危险度由该路点周围一定范围内障碍物所占面积的比例构成
    risk_degrees = []
    map_bev_obs = map_bev[1].astype(np.float32)
    radius_ego = np.round(1.5 / ((resolution[0] + resolution[1]) / 2)).astype(int)  # 转换为像素单位
    radius_search = np.round(radius / ((resolution[0] + resolution[1]) / 2)).astype(int)  # 转换为像素单位
    angle_ = np.deg2rad(angle)  # 转换为弧度

    for k, point in enumerate(path_points):
        x_id, y_id = point[:2].astype(int)
        # 将carla坐标系下的偏航转换为map_bev坐标系下，需要判断车辆是否在前进
        yaw = -(point[4] - np.pi)
        if k == len(path_points) - 1 or check_0_to_1(path_points[k, 2:], path_points[k + 1, 2:], returm_yaw=False):
            pass
        else:
            yaw = yaw + np.pi

        # 生成局部搜索矩阵和局部权重矩阵
        search_matrix, weight_matrix = generate_search_matrix_and_weight_matrix(yaw, [radius_ego, radius_search], angle_)

        # 根据xy的id，将搜索矩阵放到对应的位置上
        map_bev_search = np.zeros_like(map_bev_obs)
        map_bev_weight = np.zeros_like(map_bev_obs)
        for i in range(x_id - radius_search, x_id + radius_search + 1):
            for j in range(y_id - radius_search, y_id + radius_search + 1):
                if 0 <= i < map_bev_obs.shape[0] and 0 <= j < map_bev_obs.shape[1]:
                    map_bev_search[i, j] = search_matrix[i - x_id + radius_search, j - y_id + radius_search]
                    map_bev_weight[i, j] = weight_matrix[i - x_id + radius_search, j - y_id + radius_search]

        # 获取搜索区域范围内的障碍物
        map_bev_obs_search = map_bev_obs * map_bev_search

        # 计算危险度
        risk_degree = (map_bev_obs_search * map_bev_weight).sum()
        risk_degrees.append(risk_degree)

    risk_degrees = np.array(risk_degrees).reshape(-1, 1)

    return risk_degrees

# 将深度图中的每个像素点转换为深度
def image_depth_pixel_to_meters(image_depth):
    assert image_depth.shape[-1] == 3 and len(image_depth.shape) == 3, 'image_depth shape error'
    assert image_depth.max() <= 255.0 and image_depth.min() >= 0.0, 'image_depth value error'

    normalized = np.dot(image_depth, [1.0, 256.0, 65536.0])
    normalized /= (256 * 256 * 256 - 1)
    in_meters = 1000.0 * normalized
    return in_meters

# 将从深度图中提取label
def get_depth_label(gt_depths, down_sample_factor, d_range, depth_channels):
    flag_np2tensor = False
    if isinstance(gt_depths, np.ndarray):
        gt_depths = torch.from_numpy(gt_depths)
        flag_np2tensor = True

    B, N, H, W = gt_depths.shape
    dH, dW = H // down_sample_factor, W // down_sample_factor

    gt_depths = gt_depths.view(
        B, N, 
        dH, down_sample_factor,
        dW, down_sample_factor
    )
    gt_depths = gt_depths.permute(0, 1, 2, 4, 3, 5).contiguous()
    gt_depths = gt_depths.view(B, N, dH, dW, -1)

    # 过滤无效（0）深度，取最小非零值作为有效深度
    valid_mask = gt_depths > 0.0
    gt_depths[~valid_mask] = 1e6
    min_depth = torch.min(gt_depths, dim=-1).values
    min_depth[min_depth == 1e6] = -1.0  # 无效像素设为 -1

    # 离散化：depth -> class
    depth_labels = torch.round((min_depth - d_range[0]) / d_range[2]).long()
    depth_labels = torch.where(
        (depth_labels >= 0) & (depth_labels < depth_channels),
        depth_labels,
        torch.tensor(-1, device=depth_labels.device)  # 无效标签
    )

    if flag_np2tensor:
        depth_labels = depth_labels.numpy()
    return depth_labels

# 从每个时间戳的datas为segmentation提取数据
def extract_data_for_perception_from_datas(datas, cfgs: dict, dataset=None):
    keys = {
        'stamp', 
        'start_xyt', 'end_xyt', 'start_orientation', 'start_point_center_token', 'start_point', 'start_point_token',
        'aim_point', 'aim_point_token', 'aim_parking_plot_bev', 'aim_parking_plot_id',
    }
    if cfgs['is_rsu']:
        keys.update({
            'rsu_image', 'rsu_image_depth',  'rsu_image_raw', 'rsu_post_tran', 'rsu_post_rot', 'rsu_intrinsic', 'rsu_extrinsic', 'rsu_segmentation',
        })
    else:
        keys.update({
            'obu_image', 'obu_image_depth', 'obu_image_raw', 'obu_post_tran', 'obu_post_rot', 'obu_intrinsic', 'obu_extrinsic', 'obu_segmentation',
        })
    if dataset is not None:
        assert keys.issubset(set(dataset.keys())), 'dataset keys error'
    else:
        dataset = {k: [] for k in keys}

    # 分离部分数据
    datas_stamp = datas['stamp']
    datas_camera = datas['camera']
    datas_bev = datas['bev']
    datas_aim = datas['aim']
    
    # 提取时间戳数据
    stamp = datas_stamp['global']

    if cfgs['is_rsu']:
        # 提取路端相机数据并整合
        rsu_rgb = datas_camera['rsu_rgb']
        rsu_depth = datas_camera['rsu_depth']
        rsu_image = rsu_rgb['image_crop'][None, ...]
        rsu_image_raw = rsu_rgb['image'][None, ...]
        rsu_image_depth = image_depth_pixel_to_meters(rsu_depth['image_crop'].transpose(1, 2, 0))[None, ...]
        rsu_post_tran = rsu_rgb['post_tran'][None, ...]
        rsu_post_rot = rsu_rgb['post_rot'][None, ...]
        rsu_intrinsic = rsu_rgb['intrinsic'][None, ...]
        rsu_extrinsic = rsu_rgb['extrinsic'][None, ...]
        # 提取路端bev的数据
        rsu_segmentation = datas_bev['map_bev'][:3]
    else:
        # 提取车端相机数据并整合
        obu_image, obu_image_raw, obu_image_depth, obu_post_tran, obu_post_rot, obu_intrinsic, obu_extrinsic = [], [], [], [], [], [], []
        for pose in ['front', 'left', 'right', 'rear']:
            _obu_rgb = datas_camera[f'obu_{pose}_rgb']
            _obu_depth = datas_camera[f'obu_{pose}_depth']
            _obu_image = _obu_rgb['image_crop'][None, ...]
            _obu_image_raw = _obu_rgb['image'][None, ...]
            _obu_image_depth = image_depth_pixel_to_meters(_obu_depth['image_crop'].transpose(1, 2, 0))[None, ...]
            _obu_post_tran = _obu_rgb['post_tran'][None, ...]
            _obu_post_rot = _obu_rgb['post_rot'][None, ...]
            _obu_intrinsic = _obu_rgb['intrinsic'][None, ...]
            _obu_extrinsic = _obu_rgb['extrinsic'][None, ...]
            obu_image.append(_obu_image)
            obu_image_raw.append(_obu_image_raw)
            obu_image_depth.append(_obu_image_depth)
            obu_post_tran.append(_obu_post_tran)
            obu_post_rot.append(_obu_post_rot)
            obu_intrinsic.append(_obu_intrinsic)
            obu_extrinsic.append(_obu_extrinsic)
        obu_image = np.concatenate(obu_image, axis=0)
        obu_image_raw = np.concatenate(obu_image_raw, axis=0)
        obu_image_depth = np.concatenate(obu_image_depth, axis=0)
        obu_post_tran = np.concatenate(obu_post_tran, axis=0)
        obu_post_rot = np.concatenate(obu_post_rot, axis=0)
        obu_intrinsic = np.concatenate(obu_intrinsic, axis=0)
        obu_extrinsic = np.concatenate(obu_extrinsic, axis=0)
        # 提取车端bev的数据
        obu_segmentation = datas_bev['map_bev_obu'][:3]

    # 提取ego中心点的坐标
    ego_bev = datas_bev['map_bev'][2]
    start_point_center = get_position_id_from_bev(ego_bev)
    start_point_center_token = generate_token_from_xy(start_point_center, flag_cat=True)

    # 提取始末状态数据
    start_xyt = datas_aim['start_xyt'].reshape(1, -1)  # 初始状态车辆的位置和朝向
    end_xyt = datas_aim['end_xyt'].reshape(1, -1)  # 目标状态车辆的位置和朝向
    start_orientation = datas_aim['start_xyt'][2].reshape(1, 1)  # 初始状态车辆的朝向
    start_point = datas_aim['start_id'].reshape(1, 2)  # 起始车位的xy坐标
    start_point_token = generate_token_from_xy(start_point, flag_cat=True)  # 将起始车位的id转换为token，token的范围分别是[2, fH + 1]和[2, fW + 1]
    aim_point = datas_aim['end_id'].reshape(1, 2)  # 目标车位的xy坐标
    aim_point_token = generate_token_from_xy(aim_point, flag_cat=True)  # 将目标车位的id转换为token，token的范围分别是[2, fH + 1]和[2, fW + 1]
    aim_parking_plot_bev = datas_bev['map_bev'][3:]  # 目标车位的parking plot的bev图
    # 目标车位的parking plot的id
    aim_parking_plot_id = datas_aim['parking_plot_id'] - 9
    assert aim_parking_plot_id in [0, 1, 2], 'Unsupported aim_parking_plot_id'
    aim_parking_plot_id = np.array(aim_parking_plot_id).reshape(1)

    # 把提取的数据放入dataset中
    dtype = cfgs['dtype_model']
    dataset['stamp'].append(stamp)
    if cfgs['is_rsu']:
        dataset['rsu_image'].append(rsu_image.astype(dtype))
        dataset['rsu_image_depth'].append(rsu_image_depth.astype(dtype))
        dataset['rsu_image_raw'].append(rsu_image_raw)
        dataset['rsu_post_tran'].append(rsu_post_tran.astype(dtype))
        dataset['rsu_post_rot'].append(rsu_post_rot.astype(dtype))
        dataset['rsu_intrinsic'].append(rsu_intrinsic.astype(dtype))
        dataset['rsu_extrinsic'].append(rsu_extrinsic.astype(dtype))
        dataset['rsu_segmentation'].append(rsu_segmentation.astype(dtype))
    else:
        dataset['obu_image'].append(obu_image.astype(dtype))
        dataset['obu_image_depth'].append(obu_image_depth.astype(dtype))
        dataset['obu_image_raw'].append(obu_image_raw)
        dataset['obu_post_tran'].append(obu_post_tran.astype(dtype))
        dataset['obu_post_rot'].append(obu_post_rot.astype(dtype))
        dataset['obu_intrinsic'].append(obu_intrinsic.astype(dtype))
        dataset['obu_extrinsic'].append(obu_extrinsic.astype(dtype))
        dataset['obu_segmentation'].append(obu_segmentation.astype(dtype))
    dataset['start_xyt'].append(start_xyt.astype(dtype))
    dataset['end_xyt'].append(end_xyt.astype(dtype))
    dataset['start_orientation'].append(start_orientation.astype(dtype))
    dataset['start_point_center_token'].append(start_point_center_token.astype(dtype))
    dataset['start_point'].append(start_point.astype(dtype))
    dataset['start_point_token'].append(start_point_token.astype(dtype))
    dataset['aim_point'].append(aim_point.astype(dtype))
    dataset['aim_point_token'].append(aim_point_token.astype(dtype))
    dataset['aim_parking_plot_bev'].append(aim_parking_plot_bev.astype(dtype))
    dataset['aim_parking_plot_id'].append(aim_parking_plot_id.astype(dtype))

    return dataset

# 从每个时间戳的datas为path planning提取数据
def extract_data_for_path_planning_from_datas(datas, cfgs: dict, dataset=None):
    keys = {
        'stamp', 
        'start_xyt', 'end_xyt', 'start_orientation', 'start_point_center_token', 'start_point', 'start_point_token',
        'aim_point', 'aim_point_token', 'aim_parking_plot_bev', 'aim_parking_plot_id',
        'effective_length', 'path_point', 'path_point_token', 'planning_start_token',
        'heuristic_fig', 'risk_degree', 'planning_duration'
    }
    if cfgs['is_rsu']:
        keys.update({
            'rsu_image', 'rsu_image_depth',  'rsu_image_raw', 'rsu_post_tran', 'rsu_post_rot', 'rsu_intrinsic', 'rsu_extrinsic', 'rsu_segmentation',
        })
    else:
        keys.update({
            'obu_image', 'obu_image_depth', 'obu_image_raw', 'obu_post_tran', 'obu_post_rot', 'obu_intrinsic', 'obu_extrinsic', 'obu_segmentation',
        })
    if dataset is not None:
        assert keys.issubset(set(dataset.keys())), 'dataset keys error'
    else:
        dataset = {k: [] for k in keys}

    dataset = extract_data_for_perception_from_datas(datas, cfgs, dataset)
    
    # 分离部分数据
    datas_path = datas.get('path', None)
    datas_aim = datas['aim']

    # 提取路径数据
    if datas_path and datas_path['success']:
        # 进行拼接，并编码成一个固定维度的矩阵
        def encode_data(data, end_value, pad_value):
            data_ = np.ones([cfgs['max_num_for_path'], *data.shape[1:]]) * pad_value
            data_[0:effective_length] = data  # [0, effective_length - 1]赋值为data
            data_[effective_length] = end_value  # 结束规划之后的标志位赋值为end_value
            return data_

        path_point_ = datas_path['path_points_rear'][:, :2]  # 获取规划点的id
        # assert np.linalg.norm(path_point_[0] - datas_aim['start_id']) == 0
        assert cfgs['max_num_for_path'] >= (path_point_.shape[0] + 1), f'Need at least {path_point_.shape[0] + 1}. Get {cfgs["max_num_for_path"]}'  # +1是为了放入end_value
        path_point_token_ = generate_token_from_xy(path_point_[:, :2], flag_cat=True)  # 将规划点的id转换为token，token的范围分别是[2, fH + 1]和[2, fW + 1]
        assert not np.any(np.equal(path_point_token_, cfgs['end_value_for_path_point_token'])), 'path_point_token_ should not contain end_value'
        effective_length = path_point_.shape[0]  # 获取规划点的数量

        path_point_token = encode_data(path_point_token_, cfgs['end_value_for_path_point_token'], cfgs['pad_value_for_path_point_token'])  # end_value: 1, pad_value: 0
        path_point = np.concatenate(get_xy_id_from_token(path_point_token), axis=1)  # 反向变换，得到规划点的xy坐标，正反验证
        
        assert (path_point[:effective_length] == path_point_).all(), 'Function [get_xy_id_from_token] and [generate_token_from_xy] error'

        # 获取启发图
        heuristic_fig = datas_path['heuristic_fig']

        # 提取风险评估
        risk_degree = datas_path['risk_degrees'].reshape(-1)
        assert (risk_degree <= 1.0).all() and (risk_degree >= 0.0).all(), 'risk_degree should be in [0.0, 1.0]'
        risk_degree = encode_data(risk_degree, 0.0, 0.0)
    else: 
        effective_length = 0
        path_point = None
        path_point_token = None
        heuristic_fig = None
        risk_degree = None

    # 计算用于规划开始的token
    planning_start_token = np.ones([1, 2]) * cfgs['start_value_for_path_point_token']
    # planning_start_token = dataset['aim_point_token'][-1].copy()

    # 规划时间
    planning_duration = np.array([datas_path['duration']])

    # 把提取的数据放入dataset中
    dtype = cfgs['dtype_model']
    dataset['effective_length'].append(effective_length)
    dataset['path_point'].append(path_point.astype(dtype) if path_point is not None else None)
    dataset['path_point_token'].append(path_point_token.astype(dtype) if path_point_token is not None else None)
    dataset['heuristic_fig'].append(heuristic_fig.astype(dtype) if heuristic_fig is not None else None)
    dataset['risk_degree'].append(risk_degree.astype(dtype) if risk_degree is not None else None)
    dataset['planning_start_token'].append(planning_start_token.astype(dtype))
    dataset['planning_duration'].append(planning_duration.astype(dtype))

    assert len(set(len(v) for k, v in dataset.items())) == 1, 'dataset length error'

    return dataset

# 从pkl_files中读取可用的pkl
def get_available_pkl_files(path_pkl_files, mode, cfgs):
    available_pkl_files = os.listdir(path_pkl_files)
    available_pkl_files = sorted(available_pkl_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    available_pkl_files = [os.path.join(path_pkl_files, pkl_file) for pkl_file in available_pkl_files if pkl_file.endswith('.pkl')]

    assert len(available_pkl_files) >= cfgs['num_folder_max'], 'Dataset not enough, please check your data collection.'
    if mode == 'normal':
        pkl_files = available_pkl_files[:(cfgs['num_folder_max'] - cfgs['num_folder_free'])]
    else:
        pkl_files = available_pkl_files[(cfgs['num_folder_max'] - cfgs['num_folder_free']):cfgs['num_folder_max']]
    return pkl_files

# 从datas中提取数据给perception
@mem.cache
def extract_data_for_perception(mode, cfgs: dict):
    # 获取路径
    pkl_files = get_available_pkl_files(os.path.join(cfgs['path_datas'], 'pkl_files'), mode, cfgs)

    # 反方向遍历所有数据
    dataset = None
    for i in range(len(pkl_files) - 1, -1, -1):
        datas_all = read_datas_from_disk_without_cached(mode='pkl', path_datas=pkl_files[i])
        datas_seq = datas_all['sequence']

        dataset_ = None
        # 对于每一个sequence，提取数据
        for datas in datas_seq:
            # 将提取的数据记入dataset
            if datas['path']['success']:
                dataset_ = extract_data_for_perception_from_datas(datas, cfgs, dataset_)
        # 将dataset_合并到dataset中
        if dataset is None:
            dataset = dataset_
        else:
            if dataset_ is not None:
                for k, v in dataset_.items():
                    dataset[k].extend(v)
    return dataset

# 从datas中提取数据给pathplanning
@mem.cache
def extract_data_for_path_planning(mode, cfgs: dict):
    # 获取路径
    pkl_files = get_available_pkl_files(os.path.join(cfgs['path_datas'], 'pkl_files'), mode, cfgs)

    # 反方向遍历所有数据
    dataset = None
    for i in range(len(pkl_files) - 1, -1, -1):
        datas_all = read_datas_from_disk_without_cached(mode='pkl', path_datas=pkl_files[i])
        datas_seq = datas_all['sequence']

        dataset_ = None
        # 对于每一个sequence，提取数据
        for datas in datas_seq:
            # 将提取的数据记入dataset
            if mode == 'free' or datas['path']['success']:
                dataset_ = extract_data_for_path_planning_from_datas(datas, cfgs, dataset_)
        # 将dataset_合并到dataset中
        if dataset is None:
            dataset = dataset_
        else:
            if dataset_ is not None:
                for k, v in dataset_.items():
                    dataset[k].extend(v)

    return dataset

# 计算模型规划结果的有效长度
def get_effective_length_from_path_point_token(path_point_token, end_value, pad_value):
    assert isinstance(path_point_token, torch.Tensor), 'path_point_token type error'
    assert path_point_token.shape[-1] == 2 and len(path_point_token.shape) == 3, 'path_point_token shape error'
    
    effective_length = []
    for path_point_token_b in path_point_token:
        # 分x和y找出end_value出现的位置，如果没有end_value，则effective_length为path_point_token的长度
        end_value_x = ((path_point_token_b[:, 0] == end_value) | (path_point_token_b[:, 0] == pad_value)).nonzero()
        end_value_y = ((path_point_token_b[:, 1] == end_value) | (path_point_token_b[:, 1] == pad_value)).nonzero()
        if len(end_value_x) == 0 and len(end_value_y) == 0:
            l = path_point_token_b.shape[0]
        else:
            if len(end_value_x) == 0:
                l = end_value_y[0]
            elif len(end_value_y) == 0:
                l = end_value_x[0]
            else:
                l = min(end_value_x[0], end_value_y[0])
        l = max(l, 1)  # 至少要有一个有效的点
        effective_length.append(l)
    effective_length = torch.tensor(effective_length, dtype=torch.long, device=path_point_token.device)
    return effective_length

# 计算曲率
def cal_curvature(path_point, method='three_point'):
    assert isinstance(path_point, np.ndarray), 'path_points type error'
    assert path_point.shape[-1] == 2 and len(path_point.shape) == 2, 'path_points shape error'
    
    if len(path_point) < 3:
        return np.zeros(1)

    if method == 'three_point':
        # 三点滑窗：P_{i-1}, P_i, P_{i+1}
        p_prev = path_point[:-2]
        p_curr = path_point[1:-1]
        p_next = path_point[2:]

        # 各边向量
        a_vec = p_curr - p_prev
        b_vec = p_next - p_curr
        c_vec = p_next - p_prev

        # 边长
        a = np.linalg.norm(a_vec, axis=1)
        b = np.linalg.norm(b_vec, axis=1)
        c = np.linalg.norm(c_vec, axis=1)

        # 计算面积（海伦公式）
        s = (a + b + c) / 2
        # area = np.sqrt(s * (s - a) * (s - b) * (s - c) + 1e-6)
        area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))

        # 曲率 = 4A / (abc)
        curvature = 4 * area / (a * b * c + 1e-6)
    elif method == 'spline':
        s = 0.01
        k = min(3, len(path_point) - 1)
        # 参数化样条拟合（自动生成参数u）
        tck, u = splprep(path_point.T, s=s, per=False, k=k)
        
        # 计算一阶导数 (dx/du, dy/du) 和二阶导数 (d2x/du2, d2y/du2)
        du = splev(u, tck, der=1)    # 一阶导数
        d2u = splev(u, tck, der=2)  # 二阶导数
        
        dx, dy = du
        d2x, d2y = d2u
        
        # 计算曲率公式
        numerator = np.abs(dx * d2y - dy * d2x)
        denominator = (dx**2 + dy**2) ** 1.5
        curvature = numerator / (denominator + 1e-10)  # 避免除以零

        # # 参数化曲线（以弧长为参数）
        # s_total = np.zeros(len(path_point))
        # s_total[1:] = np.cumsum(np.linalg.norm(path_point[1:] - path_point[:-1], axis=1))

        # # 拟合样条曲线
        # x, y = path_point[:, 0], path_point[:, 1]
        # s = 0.01
        # fx = UnivariateSpline(s_total, x, s=s, k=min(3, len(path_point) - 1))
        # fy = UnivariateSpline(s_total, y, s=s, k=min(3, len(path_point) - 1))
        
        # # 计算导数
        # dx = fx.derivative()(s_total)
        # dy = fy.derivative()(s_total)
        # d2x = fx.derivative(2)(s_total)
        # d2y = fy.derivative(2)(s_total)

        # # 计算曲率公式
        # numerator = np.abs(dx * d2y - dy * d2x)
        # denominator = (dx**2 + dy**2) ** 1.5
        # curvature = numerator / (denominator + 1e-6)  # 避免除以零
    else:
        raise ValueError('Unsupported method')
    return curvature

def split_path_point(path_point):
    path_point = np.array(path_point)
    # 滤除重复点（相邻差值小于阈值）
    diffs = np.linalg.norm(path_point[1:] - path_point[:-1], axis=1)
    mask = diffs > 1e-6
    mask = np.concatenate([np.array([True]), mask])  # 保留第一个点
    path_point = path_point[mask]

    # 分离前进和后退
    vectors = path_point[1:] - path_point[:-1]
    vectors_normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6)
    # 用前后向量夹角判断折返
    cos_angles = np.sum(vectors_normed[1:] * vectors_normed[:-1], axis=1)
    # 若 cos(夹角) < cos(100°)，说明发生了明显折返，即疑似后退
    reverse_mask = cos_angles < np.cos(np.deg2rad(100))
    reverse_mask = np.concatenate([np.array([False]), reverse_mask, np.array([False])])
    # 分隔不同路段
    reverse_indices = np.where(reverse_mask)[0]
    paths = []
    if len(reverse_indices) == 0:
        paths.append(path_point)
    else:
        # 按不同路段分隔路径
        paths.append(path_point[:reverse_indices[0]+1])
        for i in range(len(reverse_indices) - 1):
            start_idx = reverse_indices[i]
            end_idx = reverse_indices[i+1]
            paths.append(path_point[start_idx:end_idx+1])
        paths.append(path_point[reverse_indices[-1]:])

    for pa in paths:
        assert len(pa) > 0, 'path_point is empty'

    return paths

def cal_yaws(points, yaw_0):
    assert points.shape[-1] == 2 and len(points.shape) == 2, 'points shape error'
    
    if len(points) == 1:
        yaws = np.array([yaw_0])
    elif len(points) == 2:
        # 生成二次贝塞尔曲线控制点
        xy_0, xy_1 = points
        d = np.linalg.norm(xy_1 - xy_0) / 3
        b = xy_0 + d * np.array([np.cos(yaw_0), np.sin(yaw_0)])
        nodes = np.stack([xy_0, b, xy_1], axis=1)
        curve = bezier.Curve(nodes, degree=2)
        # 获得末端点的斜率
        yaw_1 = curve.evaluate_hodograph(1)
        yaw_1 = np.arctan2(yaw_1[1], yaw_1[0]).item()
        # 调整朝向，使其与初始点的方向对齐
        if not check_angle_threshold(yaw_0, yaw_1, np.pi/2):
            yaw_1 = wrap_to_2pi(yaw_1 + np.pi)
        yaws = np.array([yaw_0, yaw_1])
    else:
        # 计算中间点的斜率
        diff = points[2:] - points[:-2]
        dx = diff[:, 0]
        dy = diff[:, 1]
        yaws = np.arctan2(dy, dx).tolist()
        for i in range(1, len(yaws)):
            if not check_angle_threshold(yaws[i-1], yaws[i], np.pi/2):
                yaws[i] = wrap_to_2pi(yaws[i] + np.pi)
        if not check_angle_threshold(yaw_0, yaws[0], np.pi/2):
            delta_theta = np.pi
        else:
            delta_theta = 0
        yaws = wrap_to_2pi(np.array(yaws) + delta_theta)

        # 计算末端点的斜率
        _, yaw_end = cal_yaws(points[-2:], yaws[-1])

        # 拼接
        yaws = np.concatenate([np.array([yaw_0]), yaws, np.array([yaw_end])])

    # 判断合法性
    assert len(yaws) == len(points), 'yaws shape error'
    # for i in range(len(yaws)-1):
    #     assert check_angle_threshold(yaws[i], yaws[i+1], np.pi/2), f'yaws error, i: {i}, yaws: {yaws}'
    return yaws


def add_yaw_to_path_points(path_points, last_yaw):
    yaws = cal_yaws(path_points, last_yaw)
    last_yaw = yaws[-1].item()
    path_points_with_yaw = np.concatenate([path_points, yaws.reshape(-1, 1)], axis=-1)
    return path_points_with_yaw, last_yaw

def check_collision(path_point, map_bev, start_xyt, end_xyt, cfg: Configuration):
    assert isinstance(path_point, np.ndarray), 'path_points type error'
    assert path_point.shape[-1] == 2 and len(path_point.shape) == 2, 'path_points shape error'
    assert isinstance(map_bev, np.ndarray), 'map_bev type error'

    # 分割路径
    paths = split_path_point(path_point)

    # 为每个路径点加上车辆朝向
    last_yaw = start_xyt[2]
    path_points_with_yaw = []
    for path_points in paths:
        path_points_with_yaw_, last_yaw = add_yaw_to_path_points(path_points, last_yaw)
        path_points_with_yaw.extend(path_points_with_yaw_.tolist())
    
    # 使用路径规划算法用到的工具函数
    _planner = Planner(cfg.map_bev['resolution'][:2], cfg.ego['base_params'], cfg.dtype_carla)
    park_map = costmap_for_bev.Map(
        _planner.cal_datas_init(map_bev, start_xyt, end_xyt, cfg.map_bev),
        _planner.config['map_discrete_size']
    )
    checker = collision_check.two_circle_checker(
        park_map, _planner.ego_vehicle, _planner.config
    )

    # 遍历所有路径点，进行check
    collision = False
    for path_point in path_points_with_yaw:
        collision = checker.check(*path_point)
        if collision:
            break
    return collision
        
# 测试某个函数的运行时间
def timefunc(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    return end - start, result

# 将局部xy的坐标值转为id
def xy_to_id(xy, resolution):
    xy = np.array(xy)
    assert xy.shape[-1] == 2, 'xy shape error'
    xy = xy.reshape(-1, 2)
    x, y = xy[:, 0], xy[:, 1]
    x_id = np.round(x / resolution[0])
    y_id = np.round(y / resolution[1])

    xy_id = np.stack([x_id, y_id], axis=-1)
    return xy_id

#支持变长路径的 Fréchet 距离函数
def frechet_distance(P, Q):
    """
    输入：
        P, Q: list of [x, y] 坐标，不要求点数相同。
    输出：
        float: Fréchet distance
    """
    P = np.array(P)
    Q = np.array(Q)
    assert P.shape[1] == Q.shape[1] == 2, "点必须是2D的"

    ca = -np.ones((len(P), len(Q)))

    def _c(ca, i, j, P, Q):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(np.array(P[0]) - np.array(Q[0]))
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(ca, i-1, 0, P, Q), np.linalg.norm(np.array(P[i]) - np.array(Q[0])))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(ca, 0, j-1, P, Q), np.linalg.norm(np.array(P[0]) - np.array(Q[j])))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(
                    _c(ca, i-1, j, P, Q),
                    _c(ca, i-1, j-1, P, Q),
                    _c(ca, i, j-1, P, Q)
                ),
                np.linalg.norm(np.array(P[i]) - np.array(Q[j]))
            )
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    dis = _c(ca, len(P)-1, len(Q)-1, P, Q)

    return dis

# 找出pd的置信区间对应的数
def find_credible_interval(pd, alpha=0.95):
    assert 0 < alpha < 1, 'alpha must be in (0, 1)'
    assert pd.ndim == 1, 'pd must be 1-dimensional'
    pd = np.array(pd)
    n = len(pd)
    min_len = n + 1
    best_interval = (0, n)

    for start in range(n):
        acc = 0.0
        for end in range(start + 1, n + 1):
            acc = np.sum(pd[start:end])
            if acc >= alpha:
                if (end - start) < min_len:
                    min_len = end - start
                    best_interval = (start, end)
                break  # 一旦找到满足条件的，就跳出（最短优先）

    return best_interval