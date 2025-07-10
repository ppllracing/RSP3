'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-11
FilePath: /Automated Valet Parking/map/costmap.py
Description: generate cost map

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


'''
thanks Bai Li provides the vehicle data and the map data: https://github.com/libai1943/TPCAP_demo_Python
BSD 2-Clause License

Copyright (c) 2022, Bai Li
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''




import string
import numpy as np
import math
import csv
import shapely.geometry
import matplotlib.pyplot as plt
class Vehicle:
    def __init__(self, vehicle_params: dict):
        # self.lw = vehicle_params.get('wheelbase', 2.9)  # wheelbase
        # self.lf = vehicle_params.get('front_hang_length', 0.96)  # front hang length
        # self.lr = vehicle_params.get('rear_hang_length', 0.929)  # rear hang length
        # self.lb = vehicle_params.get('width', 2.2)   # width
        # self.max_steering_angle = vehicle_params.get('max_steering_angle', 0.75)  # rad
        # self.max_angular_velocity = vehicle_params.get('max_angular_velocity', 0.5)  # rad/s
        # self.max_acc = vehicle_params.get('max_acc', 1)  # m/s^2
        # self.max_v = vehicle_params.get('max_v', 2.5)  # m/s
        # self.min_v = vehicle_params.get('min_v', -2.5)  # m/s
        self.lw = vehicle_params['wheelbase']  # wheelbase
        self.lf = vehicle_params['front_hang_length']  # front hang length
        self.lr = vehicle_params['rear_hang_length']  # rear hang length
        self.lb = vehicle_params['width']   # width
        self.max_steering_angle = vehicle_params['max_steering_angle']  # rad
        self.max_angular_velocity = vehicle_params['max_angular_velocity']  # rad/s
        self.max_acc = vehicle_params['max_acc']  # m/s^2
        self.max_v = vehicle_params['max_v']  # m/s
        self.min_v = vehicle_params['min_v']  # m/s
        self.min_radius_turn = self.lw / \
            np.tan(self.max_steering_angle) + self.lb / 2  # m

    def create_polygon(self, x, y, theta):#主要用于生成车辆在二维平面上的多边形表示 x,y为后轴中心坐标
        '''
        right back, right front, left front, left back, right back 形成闭环
        '''
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # 车辆坐标系遵循右手坐标系
        points = np.array([
            [-self.lr, -self.lb / 2, 1],
            [self.lf + self.lw, -self.lb / 2, 1],
            [self.lf + self.lw, self.lb / 2, 1],
            [-self.lr, self.lb / 2, 1],
            [-self.lr, -self.lb / 2, 1],
        ]).dot(np.array([
            [cos_theta, -sin_theta, x],
            [sin_theta, cos_theta, y],
            [0, 0, 1]
        ]).transpose())#将局部坐标点转移到全局坐标系
        return points[:, 0:2]#范围x,y

    def create_anticlockpoint(self, x, y, theta, config: dict = None):
        '''
        Note: this function will expand this vehicle square #增加安全边界
        '''
        # transform matrix
        trans_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                 [-np.sin(theta), np.cos(theta)]])

        # local point and expand this square
        # expand the collision check box
        #其实只是把这个长方体变得更大了一点
        side_dis = config['safe_side_dis']  # m
        fr_dis = config['safe_fr_dis']  # m
        right_rear = np.array([[-self.lr-fr_dis], [-self.lb/2-side_dis]])
        right_front = np.array(
            [[self.lw+self.lf+fr_dis], [-self.lb/2-side_dis]])
        left_front = np.array([[self.lw+self.lf+fr_dis], [self.lb/2+side_dis]])
        left_rear = np.array([[-self.lr-fr_dis], [self.lb/2+side_dis]])
        
        # original coordinate position
        # inverse of trans_matrix equals to transpose of trans_matrix
        points = []
        rr_point = trans_matrix.transpose().dot(
            right_rear) + np.array([[x], [y]])
        rf_point = trans_matrix.transpose().dot(
            right_front) + np.array([[x], [y]])
        lf_point = trans_matrix.transpose().dot(
            left_front) + np.array([[x], [y]])
        lr_point = trans_matrix.transpose().dot(left_rear) + \
            np.array([[x], [y]])

        points.append([rr_point[0], rr_point[1]])
        points.append([rf_point[0], rf_point[1]])
        points.append([lf_point[0], lf_point[1]])
        points.append([lr_point[0], lr_point[1]])
        points.append([rr_point[0], rr_point[1]])
        #最终得到全局坐标点
        return np.array(points)


class Case:
    def __init__(self):
        self.x0, self.y0, self.theta0 = 0, 0, 0
        self.xf, self.yf, self.thetaf = 0, 0, 0
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.obs_num = 0
        self.obs = np.array([])
        # self.vehicle = Vehicle()

    @staticmethod
    def read(datas_init):
        case = Case()

        # 将datas_init中对map_bev的描述转化为新的描述
        case.x0, case.y0, case.theta0 = datas_init['start_xyt']
        case.xf, case.yf, case.thetaf = datas_init['end_xyt']
        case.xmin, case.xmax, case.ymin, case.ymax = datas_init['bound_xy']
        case.obs = np.array(datas_init['obs'])
        return case

class Map:
    def __init__(self, datas_init, discrete_size: np.float64 = 0.1) -> None:
        self.discrete_size = discrete_size
        self.grid_index = None  # index of each grid
        self.cost_map = np.array([], dtype=np.float64)  # cost value
        self.map_position = np.array([], dtype=np.float64)  # (x,y) value
        self.case = Case.read(datas_init)
        # math.floor: return the largest integer not greater than x
        self.boundary = np.floor(np.array([self.case.xmin, self.case.xmax, self.case.ymin, self.case.ymax]))
        # self.detect_obstacle()
        self._discrete_x = 0
        self._discrete_y = 0
        # 获取map_bev原本的参数设定
        self.cfg_map_bev = datas_init['cfg_map_bev']
        # 获取障碍物信息
        self.detect_obstacle_edge()

    def discrete_map(self):
        '''
        param: case data is obtained from the csv file
        '''
        x_index = int((self.boundary[1] - self.boundary[0]) / self.discrete_size)  # x方向的总网格数
        y_index = int((self.boundary[3] - self.boundary[2]) / self.discrete_size)  # y方向的总网格数
        self.cost_map = np.zeros((x_index, y_index), dtype=np.float64)
        # create (x,y) position
        dx_position = np.linspace(self.boundary[0], self.boundary[1], x_index)
        dy_position = np.linspace(self.boundary[2], self.boundary[3], y_index)
        self._discrete_x = dx_position[1] - dx_position[0]
        self._discrete_y = dy_position[1] - dy_position[0]
        # the position of each point in the park map
        self.map_position = (dx_position, dy_position)
        # create grid index
        self.grid_index_max = x_index * y_index  # 总网格数

    def detect_obstacle_edge(self):
        # just consider the boundary of the obstacles
        # discrete map
        self.discrete_map()

        for obs in self.case.obs:
            points_x_index = np.where(
                (self.map_position[0] <= (obs[0] + self.cfg_map_bev['resolution'][0])) & ((obs[0] - self._discrete_x) < self.map_position[0])
            )

            points_y_index = np.where(
                (self.map_position[1] <= (obs[1] + self.cfg_map_bev['resolution'][1])) & ((obs[1] - self.cfg_map_bev['resolution'][1]) < self.map_position[1])
            )

            xx, yy = np.meshgrid(points_x_index[0], points_y_index[0])

            self.cost_map[xx.flatten(), yy.flatten()] = 255  # 给cost值一个较大的值
        
    def detect_obstacle(self):
        # discrete map
        self.discrete_map()

        for i in range(0, self.case.obs_num):
            obstacle = self.case.obs[i]
            # get the rectangle of the obstancle
            obstacle_xmin, obstacle_xmax = np.min(
                obstacle[:, 0]), np.max(obstacle[:, 0])
            obstacle_ymin, obstacle_ymax = np.min(
                obstacle[:, 1]), np.max(obstacle[:, 1])
            # find map points in the rectangle
            near_obs_x_index = np.where((self.map_position[0] >= obstacle_xmin) & (
                self.map_position[0] <= obstacle_xmax))
            near_obs_y_index = np.where((self.map_position[1] >= obstacle_ymin) & (
                self.map_position[1] <= obstacle_ymax))
            # determine the near points is in the obstacle or not
            # create polygon
            poly_shape = shapely.geometry.Polygon(obstacle)
            # generate potints
            points_x = self.map_position[0]
            points_y = self.map_position[1]
            for i in near_obs_x_index[0]:
                for j in near_obs_y_index[0]:
                    # print(points_x[i], points_y[j])
                    point = shapely.geometry.Point(points_x[i], points_y[j])
                    if poly_shape.intersects(point):
                        # point in the obstacl, set the cost = 255
                        if self.cost_map[i][j] != 255:
                            self.cost_map[i][j] = 255

        # print(self.cost_map.shape)

    def visual_cost_map(self):
        plt.figure(1)
        for i in range(len(self.map_position[0])):
            for j in range(len(self.map_position[1])):
                if self.cost_map[i][j] == 255:
                    plt.plot(
                        self.map_position[0][i], self.map_position[1][j], 'x', color='k')
        plt.xlim(self.case.xmin, self.case.xmax)
        plt.ylim(self.case.ymin, self.case.ymax)
        plt.draw()
        # plt.show()
        # print('ok')

    def visual_near_vehicle_map(self, xmin, xmax, ymin, ymax):
        plt.figure(1)
        for i in range(len(self.map_position[0])):
            for j in range(len(self.map_position[1])):
                if self.map_position[0][i] >= xmin and self.map_position[0][i] <= xmax and self.map_position[1][j] >= ymin and self.map_position[1][j] <= ymax:
                    plt.plot(
                        self.map_position[0][i], self.map_position[1][j], 'x', color='k')
        plt.xlim(self.case.xmin, self.case.xmax)
        plt.ylim(self.case.ymin, self.case.ymax)

    def convert_position_to_index(self,
                                  grid_x: np.float64,
                                  grid_y: np.float64):
        '''
        param: the upper right corner of the grid position
        return: the index of this grid, its range is from 1 to x_index*y_index 
        #同样也是利用一维来表示二维的索引（要变成唯一情况）[0,x_index*y_index-1]
        '''
        index_0 = math.floor((grid_x - self.boundary[0]) / self._discrete_x)#x的网格数
        index_1 = math.floor((self.boundary[3] - grid_y) / self._discrete_y) * (
            int((self.boundary[1] - self.boundary[0]) / self._discrete_x))#在这里加入x的信息
        return index_0 + index_1

    def convert_xy_to_id(self, x, y):
        x_id = math.floor(abs(x - self.boundary[0]) / self._discrete_x)
        y_id = math.floor(abs(y - self.boundary[2]) / self._discrete_y)
        return x_id, y_id