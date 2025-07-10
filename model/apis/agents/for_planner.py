import multiprocessing
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
from tqdm import tqdm
from PIL import Image

from .for_base import AgentBase
from ..tools.config import Configuration
from ..tools.util import (
    interp_Bezier, suppress_output_and_warnings, xy_to_id,
    convert_xyt_from_carla_to_normal_coord, wrap_to_2pi, timefunc
)
from ..tools.exception import *

path_folder_parking_learning_A_star = os.path.join(*(['/'] + os.path.dirname(__file__).split('/')[1:-3] + ['parking_learning_A_star']))

# 调用“parking_learning_A_star”下的算法
sys.path.append(path_folder_parking_learning_A_star)
from planner import Planner

class AgentPlanner(AgentBase):
    def __init__(self, cfg: Configuration, origin_xy, vehicle_params, *args, **kwargs):
        super().__init__(cfg)
        self.cfg = cfg
        self.dtype = self.cfg.dtype_carla
        self.origin_xy = np.array(origin_xy, dtype=self.dtype)  # 坐标原点（map_bev的原点在左上角）
        self.vehicle_params = vehicle_params
        self.jump_dis = self.cfg.collect['jump_dis']

        self.planner = None

        self.init_all()
    
    def init_all(self):
        self.planner = Planner(
            self.cfg.map_bev['resolution'][:2],
            self.vehicle_params,
            self.dtype
        )
        self.logger.info('Finish to Initialize Planner')

        super().init_all()
    
    def convert_xyt(self, xyt):
        # 将carla坐标系下的xyt转换为正常坐标系下的xyt
        xyt_ = convert_xyt_from_carla_to_normal_coord(xyt)
        origin_xy = convert_xyt_from_carla_to_normal_coord(self.origin_xy.tolist())

        # 将坐标系原点移动至左上角
        # 将x轴指向向下，y轴指向右（坐标轴顺时针旋转90度）
        # convert = lambda xyt: [-(xyt[1] - origin_xy[1]), xyt[0] - origin_xy[0], wrap_to_2pi(xyt[2] + np.pi/2)]
        xyt_ = [-(xyt_[1] - origin_xy[1]), xyt_[0] - origin_xy[0], wrap_to_2pi(xyt_[2] + np.pi/2)]

        return xyt_

    def save_for_debug(self, map_bev, start_xyt, end_xyt):
        # 保存路径规划模块所需要的输入，用于debug
        import pickle
        with open('for_input.pkl', 'wb') as f:
            pickle.dump({
                'map_bev': map_bev,
                'start_xyt': start_xyt,
                'end_xyt': end_xyt,
                'cfg_map_bev': self.cfg.map_bev
            }, f)

    # 坐标变换
    def convert_inv(self, path_points):
        path_points[:, 2] = -path_points[:, 2]  # 规划的时候是以左上角为原点，所以这里要取负值
        path_points[:, 2:4] += self.origin_xy
        path_points[:, 4] = -wrap_to_2pi(path_points[:, 4] - np.pi)  # 从x轴向下变为x轴向上，且正方向有变
        return path_points

    def cal_id(self, xyt):
        return [np.round(xyt[0] / self.cfg.map_bev['resolution'][0]), np.round(xyt[1] / self.cfg.map_bev['resolution'][1])]

    def plan(self, map_bev, start_xyt, end_xyt):
        # 进行路径规划
        # path_points_list = self.planner.run(map_bev, start_xyt, end_xyt, self.cfg.map_bev)
        with suppress_output_and_warnings():
            path_points_list = self.planner.run(map_bev, start_xyt, end_xyt, self.cfg.map_bev)

        path_points_info = {
            'success': True,
            'rear': [],
            'center': []
        }
        for path_points in path_points_list:
            path_points = np.array(path_points, dtype=self.dtype)

            path_points = self.interp_jump(path_points, cut_head=False)  # 插值和跳点

            path_points_rear = np.array(path_points, dtype=self.dtype)
            path_points_center = np.array(path_points, dtype=self.dtype)

            # 根据后轴中心点计算车辆中心的坐标
            dis_rear_center = self.vehicle_params['wheelbase'] / 2
            path_points_center[:, 0] += dis_rear_center * np.cos(path_points_center[:, 2])
            path_points_center[:, 1] += dis_rear_center * np.sin(path_points_center[:, 2])

            # 加入坐标id
            path_points_rear = self.add_grid_id_to_path_points(path_points_rear, xy_from_grid=True)
            path_points_center = self.add_grid_id_to_path_points(path_points_center, xy_from_grid=True)

            path_points_info['rear'].append(path_points_rear)
            path_points_info['center'].append(path_points_center)

        # 将path point拼起来
        path_points_info['rear'] = np.concatenate(path_points_info['rear'], axis=0)
        path_points_info['center'] = np.concatenate(path_points_info['center'], axis=0)

        # 限制路点的数量，太长的话，基本上是很抽象的轨迹
        if (len(path_points_info['rear']) + 1) > self.cfg.max_num_for_path:
            raise PlanningPathTooLongException()

        path_points_info['rear'] = self.convert_inv(path_points_info['rear'])
        path_points_info['center'] = self.convert_inv(path_points_info['center'])

        return path_points_info

    def plan_with_time_limit(self, map_bev, start_xyt, end_xyt, time_limit, raise_exception):
        def func(map_bev, start_xyt, end_xyt):
            # 通过多进程的Pool来进行时间管理
            with multiprocessing.Pool(1) as pool:
                oups = pool.apply_async(func=self.plan, args=(map_bev, start_xyt, end_xyt))
                path_points_info = None
                for i in tqdm(range(time_limit), desc='Waiting', leave=False):
                    try:
                        path_points_info = oups.get(timeout=1.0)
                        break
                    except Exception as e:
                        if isinstance(e, multiprocessing.context.TimeoutError):
                            # 此处的超时并不是需要关注的
                            continue
                        elif isinstance(e, PlanningPathTooLongException):
                            self.logger.debug('Planning Path Too Long')
                            if raise_exception:
                                raise e
                            break
                        elif isinstance(e, PlanningPathOutOfBoundException):
                            self.logger.debug('Planning Path Out of Bound')
                            if raise_exception:
                                raise e
                            break
                        else:
                            print(traceback.format_exc())
                            # 代码错误直接反馈
                            assert False, 'Raise an Unexcepted Exception. Maybe Code Error'
                if path_points_info is None and i == time_limit - 1:
                    # 超时，且没有得到结果
                    self.logger.debug('Planning Out of Time')
                    if raise_exception:
                        raise PlanningOutOfTimeException()
            return path_points_info
        duration, path_points_info = timefunc(func, map_bev, start_xyt, end_xyt)

        if path_points_info is None:
            # 计算初始点的坐标
            path_point_rear = np.array(start_xyt, dtype=self.dtype).reshape(1, -1)
            path_point_center = np.array(start_xyt, dtype=self.dtype).reshape(1, -1)
            # 根据后轴中心点计算车辆中心的坐标
            dis_rear_center = self.vehicle_params['wheelbase'] / 2
            path_point_center[:, 0] += dis_rear_center * np.cos(path_point_center[:, 2])
            path_point_center[:, 1] += dis_rear_center * np.sin(path_point_center[:, 2])

            # 加入坐标id
            path_point_rear = self.add_grid_id_to_path_points(path_point_rear, xy_from_grid=False)
            path_point_rear = self.convert_inv(path_point_rear)
            path_point_center = self.add_grid_id_to_path_points(path_point_center, xy_from_grid=False)
            path_point_center = self.convert_inv(path_point_center)
            path_points_info = {
                'success': False,
                'rear': path_point_rear,
                'center': path_point_center
            }
        
        path_points_info.update({'duration': duration})
        return path_points_info

    # 插值和跳点
    def interp_jump(self, xyt, cut_head=True):
        xyt = np.array(xyt)

        # 对点进行插值
        xyt_interp = [xyt[0]]
        for i in range(1, len(xyt)):
            xyt_cur = xyt_interp[-1]
            xyt_aim = xyt[i]
            # 插值
            xyt_interp.extend(interp_Bezier(xyt_cur, xyt_aim))
        xyt_interp = np.stack(xyt_interp)

        # 进行跳点
        dis = lambda xyt_1, xyt_2: np.sqrt(np.sum((xyt_1[:2] - xyt_2[:2])**2))
        select_id = [0]
        for i in range(len(xyt_interp) - 1):
            if dis(xyt_interp[i], xyt_interp[select_id[-1]]) >= self.jump_dis:
                select_id.append(i)

        # 判断如何处理尾部点
        if dis(xyt_interp[-1], xyt_interp[select_id[-1]]) > 0.5 * self.jump_dis or len(select_id) == 1:
            select_id.append(len(xyt_interp) - 1)
        else:
            select_id[-1] = len(xyt_interp) - 1

        # 跳点
        xyt_oup = xyt_interp[select_id]

        if cut_head:
            # 跳点，并去头
            xyt_oup = xyt_oup[1:]

        return xyt_oup

    # 将xy坐标转换为网格图中的坐标
    def add_grid_id_to_path_points(self, path_xyt, xy_from_grid=True):
        # 将x,y转为网格图中的坐标
        x, y, t = np.array(path_xyt, dtype=self.dtype).reshape(-1, 3).T
        xy_id = xy_to_id(np.stack([x, y], axis=-1), self.cfg.map_bev['resolution'])
        x_ids, y_ids = xy_id.T

        if not ((self.cfg.map_bev['final_dim'][0] > x_ids).all() and (x_ids >= 0).all() and (self.cfg.map_bev['final_dim'][1] > y_ids).all() and (y_ids >= 0).all()):
            # 出现种情况的愿意，基本上都是规划算法失败的情况
            # 经过分析，如下会出现如此报错
            # 1. 在起点进行RS曲线规划的时候，就出现了超出边界的路径点
            # 2. 在此基础上，进行探索的时候，找不到合适的下一个点，可能都会碰撞
            raise PlanningPathOutOfBoundException()

        # 输出重组
        if xy_from_grid:
            # 根据id重新计算[x,y]，因为[x, y]和[x_, y_]的距离差别并不大，所以t不作重新计算
            x_ = x_ids * self.cfg.map_bev['resolution'][0]
            y_ = y_ids * self.cfg.map_bev['resolution'][1]
            xyt_oup = np.stack([x_ids, y_ids, x_, y_, t], axis=-1)
        else:
            xyt_oup = np.stack([x_ids, y_ids, x, y, t], axis=-1)

        return xyt_oup