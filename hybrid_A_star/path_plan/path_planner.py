'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-08
FilePath: /Automated Valet Parking/path_plan/path_planner.py
Description: path plan

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


import copy
import queue
import numpy as np
from scipy import spatial
from typing import Dict, Tuple, List
from PIL import Image
from tqdm import tqdm

from path_plan.hybrid_a_star import hybrid_a_star, Node
from animation.animation import ploter
from map.costmap import Vehicle
from map.costmap_for_bev import Map
from collision_check import collision_check
from path_plan.rs_curve import PATH
from path_plan.compute_h import Dijkstra


class PathPlanner:
    def __init__(self,
                 config: dict = None,
                 map: Map = None,
                 vehicle: Vehicle = None) -> None:
        self.config = config
        self.map = map
        self.vehicle = vehicle
        if config['collision_check'] == 'circle':
            self.collision_checker = collision_check.two_circle_checker(
                map=map,
                vehicle=vehicle,
                config=config
            )
        else:
            self.collision_checker = collision_check.distance_checker(
                map=map,
                vehicle=vehicle,
                config=config
            )

        self.planner = hybrid_a_star(config=config, park_map=map, vehicle=vehicle)

    def path_planning(self) -> Tuple[List[List], Dict, List[List[List]]]:
        # final_path由两部分拼接而成，第一部分是astar_path，第二部分是rs_path
        final_path, astar_path, rs_path = self.a_star_plan()   

        # for x, y, theta in astar_path:
        #     if self.collision_checker.check(node_x=x, node_y=y, theta=theta):
        #         print("collision in astar_path")
        # for x, y, theta in zip(rs_path.x, rs_path.y, rs_path.yaw):
        #     if self.collision_checker.check(node_x=x, node_y=y, theta=theta):
        #         print("collision in rs_path")
        # for x, y, theta in final_path:
        #     if self.collision_checker.check(node_x=x, node_y=y, theta=theta):
        #         print("collision in final_path")

        # 由于final_path中可能存在前进的路端，此处将其按照是否换挡进行分割
        path_points_list, _ = self.split_path(final_path)
        path_points_list = [np.array(path_points) for path_points in path_points_list]

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # final_path_ = np.array(final_path)
        # plt.plot(final_path_[:, 0], final_path_[:, 1], 'r-')

        # x, y, theta = path_points_list[1][-1]
        # plt.plot(x, y, 'ro')
        # v = self.vehicle
        # side_dis = self.config['safe_side_dis']  # m
        # fr_dis = self.config['safe_fr_dis']  # m
        # # compute circle diameter
        # Rd = 0.5 * np.sqrt(((v.lr+v.lw+v.lf)/2)**2 + (v.lb**2)) + max(side_dis, fr_dis)
        # # compute circle center position
        # front_circle = (x+1/4*(3*v.lw+3*v.lf-v.lr)*np.cos(theta),
        #                 y+1/4*(3*v.lw+3*v.lf-v.lr)*np.sin(theta))
        # rear_circle = (x+1/4*(v.lw+v.lf-3*v.lr)*np.cos(theta),
        #                y+1/4*(v.lw+v.lf-3*v.lr)*np.sin(theta))
        # theta_ = np.linspace(0, 2*np.pi, 100)
        # x_ = Rd * np.cos(theta_)
        # y_ = Rd * np.sin(theta_)
        # plt.plot(front_circle[0] + x_, front_circle[1] + y_, 'b')
        # plt.plot(front_circle[0], front_circle[1], 'bo')
        # plt.plot(rear_circle[0] + x_, rear_circle[1] + y_, 'g')
        # plt.plot(rear_circle[0], rear_circle[1], 'go')
        # xy = np.where(self.map.cost_map == 255)
        # for x, y in zip(xy[0], xy[1]):
        #     plt.plot(x * 0.2, y * 0.2, 'k*')     
        # plt.gca().set_aspect('equal', adjustable='box')

        return path_points_list

    def cal_heuristic_fig(self, datas_init: dict, final_dim: List, flag_normalize=True) -> np.ndarray:
        # 根据datas_init生成新的map
        new_map = Map(datas_init, self.config['map_discrete_size'])

        # 将新的map传入到启发函数中
        heuristic_handle = Dijkstra(new_map)
        _, h_value_list = heuristic_handle.compute_path(*datas_init['start_xyt'][:2])

        # 获取当前状态下的启发
        heuristic_fig = np.zeros_like(self.map.cost_map)
        for h_value in h_value_list:
            x_id, y_id = self.map.convert_xy_to_id(h_value.grid_x, h_value.grid_y)
            heuristic_fig[x_id, y_id] = h_value.distance

        # 归一化
        if flag_normalize:
            heuristic_fig = (heuristic_fig - heuristic_fig.min()) / (heuristic_fig.max() - heuristic_fig.min())

        # 调整尺寸
        heuristic_fig_ = Image.fromarray((heuristic_fig * 255).astype(np.uint8))
        heuristic_fig_reshaped = heuristic_fig_.resize([final_dim[1], final_dim[0]], Image.Resampling.BICUBIC)
        heuristic_fig_reshaped = np.array(heuristic_fig_reshaped) / 255

        return heuristic_fig_reshaped

    def a_star_plan(self) -> Tuple[List[List], List[List], PATH]:
        '''
        use a star to search a feasible path and use rs curve to reach the goal,
        final_path = astar_path + rs_path
        return: final_path, astar_path, rs_path
        '''
        astar = self.planner
        # 在一开始的时候，astar.open_list.qsize() = 1，即只包含起点

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.gca().set_aspect('equal', adjustable='box')
        # xy = np.where(self.map.cost_map == 255)
        # for x, y in zip(xy[0], xy[1]):
        #     plt.plot(x * 0.2, y * 0.2, 'k*')     

        # 开始搜索
        while not astar.open_list.empty():
            # get current node
            # 获取当前所有点中，优先级最高的点
            current_node = astar.open_list.get()

            # v = self.vehicle
            # x, y, theta = current_node.x, current_node.y, current_node.theta
            # # compute circle diameter
            # Rd_min = 0.5 * np.sqrt(((v.lr+v.lw+v.lf)/2)**2 + (v.lb**2))
            # Rd_max = (v.lr+v.lw+v.lf)/4
            # Rd = (Rd_min + Rd_max) / 2
            # # compute circle center position
            # front_circle = (x+1/4*(3*v.lw+3*v.lf-v.lr)*np.cos(theta),
            #                 y+1/4*(3*v.lw+3*v.lf-v.lr)*np.sin(theta))
            # rear_circle = (x+1/4*(v.lw+v.lf-3*v.lr)*np.cos(theta),
            #             y+1/4*(v.lw+v.lf-3*v.lr)*np.sin(theta))
            # theta_ = np.linspace(0, 2*np.pi, 100)
            # x_ = Rd * np.cos(theta_)
            # y_ = Rd * np.sin(theta_)
            # plt.plot(front_circle[0] + x_, front_circle[1] + y_, 'b')
            # plt.plot(front_circle[0], front_circle[1], 'bo')
            # plt.plot(rear_circle[0] + x_, rear_circle[1] + y_, 'g')
            # plt.plot(rear_circle[0], rear_circle[1], 'go')

            # 生成rs路径，并检测是否发生碰撞
            rs_path, collision, info = astar.try_reach_goal(current_node)

            # plt.plot(current_node.x, current_node.y, 'ro')
            # plt.plot(rs_path.x, rs_path.y, 'g--')

            # 判断结束
            if not collision and info['in_radius']:
                break

            # 通过离散的方向盘转角来探索在下一时刻车辆可能会到达的位置，并将不会发生碰撞的位置加入open_list中
            child_group = astar.expand_node(current_node)
        a_star_path = astar.finish_path(current_node)

        final_path = copy.deepcopy(a_star_path[:-1])  # rs_path中会包含此处的终点，所以不用再加一次
        # final_path = a_star_path + rs_path
        # a_star_path：正向开入，主要是调整位置
        # rs_path：倒车入库，主要是一把进
        # assemble all path
        for x, y, theta in zip(rs_path.x, rs_path.y, rs_path.yaw):
            final_path.append([x, y, theta])

        return final_path, a_star_path, rs_path

    def split_path(self, final_path: List[List]) -> Tuple[List[List[List]], int]:
        '''
        split the final path (a star + rs path) into severial single path for optimization
        input: final_path is generated from the planner
        return: split_path, change_gear
        '''
        # split path based on the gear (forward or backward)
        split_path = []
        change_gear = 0
        start = 0
        extend_num = self.config['extended_num']
        # we want to extend node but these points also need collision check
        have_extended_points = 0

        for i in range(len(final_path) - 2):
            vector_1 = (final_path[i+1][0] - final_path[i][0],
                        final_path[i+1][1] - final_path[i][1])

            vector_2 = (final_path[i+2][0] - final_path[i+1][0],
                        final_path[i+2][1] - final_path[i+1][1])

            compute_cosin = 1 - spatial.distance.cosine(vector_1, vector_2)

            # if cosin < 0, it is a gear change
            if compute_cosin < 0:
                change_gear += 1
                end = i+2
                input_path = final_path[start:end]

                if change_gear > 1 and have_extended_points > 0:
                    # add extend node into the input path
                    pre_path = split_path[-1]
                    for j in range(have_extended_points):
                        x_j = pre_path[-(have_extended_points-j)][0]
                        y_j = pre_path[-(have_extended_points-j)][1]
                        theta_j = pre_path[-(have_extended_points-j)][2]
                        input_path.insert(0, [x_j, y_j, theta_j])

                    have_extended_points = 0

                # extend points
                for j in range(extend_num):
                    forward_1 = (final_path[i+1][0] > final_path[i][0]) and (
                        final_path[i][2] > -np.pi/2 and final_path[i][2] < np.pi/2)
                    forward_2 = (final_path[i+1][0] < final_path[i][0]) and (
                        (final_path[i][2] > np.pi/2 and final_path[i][2] < np.pi) or (final_path[i][2] > -np.pi and final_path[i][2] < -np.pi/2))
                    if forward_1 or forward_2:
                        speed = self.vehicle.max_v
                    else:
                        speed = -self.vehicle.max_v

                    td_j = speed * self.planner.ddt * (j+1)
                    theta_j = final_path[i+1][2]
                    x_j = final_path[i+1][0] + td_j * np.cos(theta_j)
                    y_j = final_path[i+1][1] + td_j * np.sin(theta_j)

                    collision = self.collision_checker.check(node_x=x_j,
                                                             node_y=y_j,
                                                             theta=theta_j)

                    if not collision:
                        input_path.append([x_j, y_j, theta_j])
                        have_extended_points += 1

                split_path.append(input_path)
                start = i+1

        # add final episode path
        input_path = final_path[start:]

        if have_extended_points > 0 and len(split_path) > 0:
            pre_path = split_path[-1]
            for j in range(have_extended_points):
                x_j = pre_path[-(have_extended_points-j)][0]
                y_j = pre_path[-(have_extended_points-j)][1]
                theta_j = pre_path[-(have_extended_points-j)][2]
                input_path.insert(0, [x_j, y_j, theta_j])

        split_path.append(input_path)

        return split_path, int(change_gear)
